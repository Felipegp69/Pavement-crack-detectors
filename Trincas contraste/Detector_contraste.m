clc; clear; close all;

%% Carregar e redimensionar imagem
I = imread('Fotos/DJI_0444.jpg');
% I = imresize(I, 2, 'bicubic');  % Aumenta a resolução (se necessário)

% Definir parâmetros
blockSize = [100, 200];  % Tamanho dos blocos
[height, width, ~] = size(I);
resultados = cell(ceil(height / blockSize(1)), ceil(width / blockSize(2)));

%% Processar cada bloco
se = strel('line', 11, 0);  % Horizontal

for row = 1:blockSize(1):height
    for col = 1:blockSize(2):width
        rowEnd = min(row + blockSize(1) - 1, height);
        colEnd = min(col + blockSize(2) - 1, width);
        block = I(row:rowEnd, col:colEnd, :);
        % figure(2), imshow(block);

        % Processamento da imagem
        Igray = rgb2gray(block);
        Istrech = adapthisteq(Igray); % Melhor ajuste de contraste
        K = medfilt2(Istrech, [1 2]); % Filtro de mediana na horizontal
        
        % Aplicar bot-hat
        bothatimg = imbothat(K, se);

        % Binarização adaptativa
        T = graythresh(bothatimg);
        BW = imbinarize(bothatimg, T);

        % Remover ruídos pequenos
        minSize = round(numel(BW) * 0.005);
        BW2 = bwareaopen(BW, minSize);

        % Afinar a segmentação
        BW3 = bwmorph(BW2, 'thin', Inf);

        % Remover componentes majoritariamente horizontais
        CC = bwconncomp(BW3);
        stats = regionprops(CC, 'BoundingBox');
        maskFiltered = false(size(BW3));

        for k = 1:CC.NumObjects
            bbox = stats(k).BoundingBox;
            width_comp = bbox(3);
            height_comp = bbox(4);

            if height_comp >= 3 * width_comp  % Manter se o comprimento for 3 vezes maior verticalmente 
                maskFiltered(CC.PixelIdxList{k}) = true;
            end
        end

        BW4 = maskFiltered;

        % Armazenar o resultado
        blockRow = ceil(row / blockSize(1));
        blockCol = ceil(col / blockSize(2));
        resultados{blockRow, blockCol} = BW4;
    end
end

% Recombinar os blocos processados
resultado_final = false(height, width);
for row = 1:blockSize(1):height
    for col = 1:blockSize(2):width
        rowEnd = min(row + blockSize(1) - 1, height);
        colEnd = min(col + blockSize(2) - 1, width);

        blockRow = ceil(row / blockSize(1));
        blockCol = ceil(col / blockSize(2));

        resultado_final(row:rowEnd, col:colEnd) = resultados{blockRow, blockCol};
    end
end

%% Mesclar imagem inicial com o resultado final
I_highlighted = I;
mascara = resultado_final;

% Destacar pixels da máscara em laranja
I_highlighted(repmat(mascara, [1, 1, 3])) = 0;
I_highlighted(:, :, 1) = I_highlighted(:, :, 1) + uint8(mascara) * 255; % Vermelho
I_highlighted(:, :, 2) = I_highlighted(:, :, 2) + uint8(mascara) * 128; % Verde

%% Mostrar resultado final
figure(3), imshow(I_highlighted);
title('Imagem com Rachaduras Verticais Destacadas em Laranja');

%% Salvar imagem final com nome incremental
outputPath = 'Resultado';
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
imwrite(I_highlighted, fullfile(outputPath, fileName));
disp(['Imagem salva com sucesso como: ' fullfile(outputPath, fileName)]);
