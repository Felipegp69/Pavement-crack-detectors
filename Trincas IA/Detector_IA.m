clc; close all;

% Caminho da imagem de entrada (na mesma pasta do script)
imagePath = fullfile(fileparts(mfilename('fullpath')), 'Fotos/DJI_0444.jpg');

% Parâmetros principais da imagem e do modelo
gsd = 0.55;             % Resolução espacial: 0,55 cm/pixel
lado_cm = 50;           % Lado do bloco em centímetros
overlapFactor = 0.25;   % Fator de sobreposição (25%)
inputSize = [227, 227]; % Tamanho da imagem de entrada exigido pela IA

% Parâmetros de detecção
scoreThreshold = 0.35;      % Score mínimo para considerar trinca
gradcamThreshold = 0.3;     % Limite para destacar regiões no Grad-CAM
alpha = 0.3;                % Transparência da sobreposição Grad-CAM

% Cálculo do tamanho do bloco em pixels
blockSize = [round(lado_cm / gsd), round(lado_cm / gsd)];
[blockH, blockW] = deal(blockSize(1), blockSize(2));

% Margem de expansão de cada bloco (para recobrimento)
padH = round(overlapFactor * blockH / 2);
padW = round(overlapFactor * blockW / 2);

% Leitura da imagem original
inputImage = imread(imagePath);
[height, width, ~] = size(inputImage);

% Cálculo do número de blocos (tamanho da malha)
nLin = ceil(height / blockH);
nCol = ceil(width / blockW);
matrizAfetada = zeros(nLin, nCol); % Armazena quais blocos contêm trincas
recombinedImage = zeros(height, width, 3, 'like', inputImage); % Imagem final

% Loop por cada bloco da malha (sem sobreposição na malha!)
for i = 1:nLin
    for j = 1:nCol
        % Coordenadas do bloco original (sem sobreposição)
        rowStart = (i - 1) * blockH + 1;
        colStart = (j - 1) * blockW + 1;
        rowEnd = min(rowStart + blockH - 1, height);
        colEnd = min(colStart + blockW - 1, width);

        % Coordenadas expandidas (com sobreposição)
        padRowStart = max(1, rowStart - padH);
        padColStart = max(1, colStart - padW);
        padRowEnd = min(height, rowEnd + padH);
        padColEnd = min(width, colEnd + padW);

        % Extrai bloco expandido da imagem
        expandedBlock = inputImage(padRowStart:padRowEnd, padColStart:padColEnd, :);

        % Ajuste de contraste e redimensionamento para entrada da IA
        Istrech = imadjust(expandedBlock, stretchlim(expandedBlock));
        resizedBlock = imresize(Istrech, inputSize);

        % Classificação com rede treinada (IA)
        [label, scores] = classify(netTransfer, resizedBlock);

        % Extrair o score associado à classe "Positive"
        classNames = netTransfer.Layers(end).Classes;
        positiveScore = scores(strcmp(string(classNames), 'Positive'));

        % Verifica se o score excede o limiar definido
        if positiveScore >= scoreThreshold
            matrizAfetada(i, j) = 1; % Marca bloco como afetado por trinca

            % Calcula mapa de ativação Grad-CAM para explicação visual
            dlImg = dlarray(single(resizedBlock), 'SSC');
            [featureMap, dScoresdMap] = dlfeval(@gradcam, dlnet, dlImg, softmaxName, featureLayerName, label);

            % Combina mapa de ativação e gradientes
            gradcamMap = sum(featureMap .* sum(dScoresdMap, [1, 2]), 3);
            gradcamMap = rescale(extractdata(gradcamMap)); % Normaliza entre 0 e 1

            % Cria máscara para destacar regiões com alta ativação
            heatmapMask = gradcamMap >= gradcamThreshold;
            heatmapRGB = ind2rgb(im2uint8(gradcamMap), jet(256)); % Converte para RGB colormap
            heatmapRGB(~heatmapMask) = 0; % Remove regiões fracas

            % Redimensiona mapa de calor para a área do bloco original
            heatmapResized = imresize(heatmapRGB, [rowEnd - rowStart + 1, colEnd - colStart + 1]);
            maskResized = imresize(heatmapMask, [rowEnd - rowStart + 1, colEnd - colStart + 1]);

            % Sobrepõe Grad-CAM ao bloco original com transparência
            originalBlock = im2double(inputImage(rowStart:rowEnd, colStart:colEnd, :));
            overlayBlock = originalBlock;
            for c = 1:3
                overlayBlock(:, :, c) = alpha * heatmapResized(:, :, c) .* maskResized + ...
                                        (1 - alpha) * originalBlock(:, :, c);
            end

            % Substitui o bloco na imagem recombinada
            recombinedImage(rowStart:rowEnd, colStart:colEnd, :) = im2uint8(overlayBlock);
        else
            % Caso negativo, mantém bloco original na imagem recombinada
            recombinedImage(rowStart:rowEnd, colStart:colEnd, :) = inputImage(rowStart:rowEnd, colStart:colEnd, :);
        end
    end
end

% Exibe a imagem final com sobreposição Grad-CAM
figure;
imshow(recombinedImage, []);
title('Imagem Recombinada com Sobreposição Local e Malha Original');

%% Salvar imagem final
outputPath = 'Resultados';
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end
fileList = dir(fullfile(outputPath, '*.jpg'));
numeros = [];
for i = 1:length(fileList)
    [~, name, ~] = fileparts(fileList(i).name);
    num = str2double(name);
    if ~isnan(num)
        numeros = [numeros, num];
    end
end
if isempty(numeros)
    novoNumero = 1;
else
    novoNumero = max(numeros) + 1;
end
fileName = sprintf('%04d.jpg', novoNumero);
imwrite(recombinedImage, fullfile(outputPath, fileName));
disp(['Imagem salva com sucesso como: ' fullfile(outputPath, fileName)]);

% Visualizar matriz afetada
outputPathMI = 'Resultados\Matriz_Imag';
MatrizIA(matrizAfetada, lado_cm, gsd, fullfile(outputPathMI, ['matriz_' fileName]));

%% Função Grad-CAM
function [featureMap,dScoresdMap] = gradcam(dlnet, dlImg, softmaxName, featureLayerName, classfn)
    [scores,featureMap] = predict(dlnet, dlImg, 'Outputs', {softmaxName, featureLayerName});
    classScore = scores(classfn);
    dScoresdMap = dlgradient(classScore,featureMap);
end

%% Função para gerar imagem da matriz binária
function MatrizIA(matriz, lado_cm, gsd, outputFileName)
    quadSize = round(lado_cm / gsd);
    separador = 8;
    [nLin, nCol] = size(matriz);
    altura = nLin * quadSize + (nLin - 1) * separador;
    largura = nCol * quadSize + (nCol - 1) * separador;
    img = uint8(255 * ones(altura, largura, 3));
    for i = 1:nLin
        for j = 1:nCol
            rowStart = (i-1) * (quadSize + separador) + 1;
            rowEnd = rowStart + quadSize - 1;
            colStart = (j-1) * (quadSize + separador) + 1;
            colEnd = colStart + quadSize - 1;
            if matriz(i,j) == 1
                cor = uint8(cat(3, 255*ones(quadSize), zeros(quadSize), zeros(quadSize)));
            else
                cor = uint8(cat(3, zeros(quadSize), zeros(quadSize), 255*ones(quadSize)));
            end
            img(rowStart:rowEnd, colStart:colEnd, :) = cor;
        end
    end
    imwrite(img, outputFileName);
    figure(2);
    imshow(img);
    title('Visualização da Matriz Afetada');
    hold on;
    h1 = plot(nan, nan, 's', 'MarkerEdgeColor', 'none', 'MarkerFaceColor', [1 0 0], 'MarkerSize', 10); % vermelho
    h2 = plot(nan, nan, 's', 'MarkerEdgeColor', 'none', 'MarkerFaceColor', [0 0 1], 'MarkerSize', 10); % azul
    legend([h1, h2], {'Seções com trincas', 'Seções sem trincas'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
    hold off;
    
    % Salvar matriz .mat
    outputPathM = 'Resultados\Matriz';
    [~, nameWithoutExt, ~] = fileparts(outputFileName);
    matPath = fullfile(outputPathM, [nameWithoutExt, '.mat']);
    save(matPath, 'matriz');
end
