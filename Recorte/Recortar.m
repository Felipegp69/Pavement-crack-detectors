clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables. Or clearvars if you want.
% workspace;  % Make sure the workspace panel is showing.
% format long g;
% format compact;
% fontSize = 16;
% markerSize = 20;

% Caminho para a pasta contendo as imagens originais (pode ser um caminho absoluto ou relativo)
pasta_entrada = 'Fotos';

% Obter lista de todas as imagens na pasta de entrada
extensoesValidas = {'*.jpg', '*.png', '*.bmp', '*.tif'}; % Extensões de imagens suportadas
arquivos = [];
for i = 1:length(extensoesValidas)
    arquivos = [arquivos; dir(fullfile(pasta_entrada, extensoesValidas{i}))];
end

% Verificar se existem imagens na pasta
if isempty(arquivos)
    error('Nenhuma imagem encontrada na pasta de entrada.');
end

% Verificar se a pasta "recortes" existe, se não, cria a pasta
pasta_recortes = 'Recortes';
if ~exist(pasta_recortes, 'dir')
    mkdir(pasta_recortes); % Cria a pasta "recortes" caso não exista
end

% ----------------------------------------------------------------------------------------------------------------------
% Recortes de imagens
for i = 1:length(arquivos)
    % Ler a imagem
    caminhoImagem = fullfile(pasta_entrada, arquivos(i).name);
    rgbImage = imread(caminhoImagem);
    
    % DIVIDIDO EM CANAIS DE CORES INDIVIDUAIS PARA QUE POSSAMOS ENCONTRAR O BRANCO.
    [redImage, greenImage, blueImage] = imsplit(rgbImage);
    threshold = 175;  % valor limear para classificação de pixel 
    whitePixels = (redImage >= threshold) & (greenImage >= threshold) & (blueImage >= threshold);
    
    % Preencha buracos
    mask = imfill(whitePixels, 'holes');
    % Considera apenas as 15 maiores regiões 
    mask = bwareafilt(mask, 15);
    % Erodi uma camada de pixel para se livrar do contorno branco.
    mask = imerode(mask, ones(3));
    % Apague a parte externa da máscara definindo esses pixels como zero.
    redImage(~mask) = 0;
    greenImage(~mask) = 0;
    blueImage(~mask) = 0;
    % Reconstroi a imagem
    rgbImage2 = cat(3, redImage, greenImage, blueImage);
    %figure(1), imshow(rgbImage2);
    %imwrite(rgbImage2, fullfile(pasta_recortes, sprintf('%02d.jpg', i)));

    % Encontrar caixa de recorte
    margemEsq = 0;
    margemDir = 0;
    [rows, columns] = find(mask);
    row1 = min(rows);
    row2 = max(rows);
    col1 = min(columns) -  margemEsq;
    col2 = max(columns) + margemDir;

    % Crop original image
    rgbImage3 = rgbImage(row1:row2, col1:col2, :);
    
    % Contar o número de arquivos .jpg na pasta "recortes"
    arquivos_existentes = dir(fullfile(pasta_recortes, '*.jpg'));
    numero_imagem = length(arquivos_existentes) + 1; % Próximo número
    
    % Formatar o número da imagem com 2 dígitos
    nome_imagem = sprintf('%02d.jpg', numero_imagem);
    
    % Salvar a imagem recortada na pasta "recortes"
    caminho_arquivo = fullfile(pasta_recortes, nome_imagem);
    imwrite(rgbImage3, caminho_arquivo);
end 
