%% Visualizzazione KUKA iiwa7 + Camera (Hand-Eye) SOLO EE + Camera (nostro caso)
clear; clc; close all;

%% === 1) Import robot iiwa7 ===
robot = importrobot('iiwa7.urdf');
robot.DataFormat = 'row';
robot.Gravity = [0 0 -9.81];

%% === 2) Configurazione giunti (esempio) ===
q = deg2rad([0, 30, 0, -60, 0, 90, 0]);

%% === 3) EE pose in WORLD ===
baseName = robot.BaseName;
eeName   = robot.BodyNames{end};   % meglio se metti il link tool reale (es. 'iiwa_link_ee' se esiste)
T_ee2world = getTransform(robot, q, eeName, baseName);  % ^worldT_ee

%% === 4) Hand-eye dal Python: Tcamee (camera -> EE) in METRI ===
% Dal tuo output:
Tcamee = [ 0.99996874 -0.00314454 -0.00725433  0.00858134;
           0.00606781  0.89344557  0.44913049 -0.17134914;
           0.00506904 -0.44916047  0.89343672  0.02954765;
           0           0           0           1];

% Interpretazione:
% - normalmente Tcamee stampato come "camera -> EE" lo usiamo come ^EE T_cam
%   quindi T_cam2ee = Tcamee
% Se ti viene specchiato, metti USE_INVERSE = true
USE_INVERSE = false;

if USE_INVERSE
    T_cam2ee = inv(Tcamee);   % alternativa
else
    T_cam2ee = Tcamee;        % ^EE T_cam
end

%% === 5) Camera pose in WORLD ===
T_cam2world = T_ee2world * T_cam2ee;   % ^worldT_cam

%% --- Plot robot (NO frame su tutti i joint) ---
figure('Color','w');
ax = show(robot, q, ...
    'PreservePlot', false, ...
    'Frames', 'off');
hold on; axis equal; grid on;

xlim([-0.2  0.8])
ylim([-0.5  0.5])
zlim([ 0.0  1.0])
view(135, 30);

xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
title('KUKA iiwa7 with EE + Camera');

camlight('headlight');
material('dull');
light('Position',[1 1 1], 'Style','infinite');

%% === Funzione helper per disegnare un frame (assi XYZ) ===
drawFrame = @(T, s, lw) ...
    [ ...
    quiver3(T(1,4),T(2,4),T(3,4), s*T(1,1),s*T(2,1),s*T(3,1), 'r','LineWidth',lw, 'MaxHeadSize',0.8), ...
    quiver3(T(1,4),T(2,4),T(3,4), s*T(1,2),s*T(2,2),s*T(3,2), 'g','LineWidth',lw, 'MaxHeadSize',0.8), ...
    quiver3(T(1,4),T(2,4),T(3,4), s*T(1,3),s*T(2,3),s*T(3,3), 'b','LineWidth',lw, 'MaxHeadSize',0.8) ...
    ];

%% --- Disegna SOLO frame EE ---
scaleEE = 0.06;
drawFrame(T_ee2world, scaleEE, 2);
ee_origin = T_ee2world(1:3,4);
text(ee_origin(1) - 0.02, ee_origin(2) -0.075, ee_origin(3) - 0.03, 'EE', ...
    'Color','k','FontSize',10,'FontWeight','bold');

%% --- Disegna SOLO camera: frame + sfera ---
scaleCam = 0.05;
drawFrame(T_cam2world, scaleCam, 2);

cam_origin = T_cam2world(1:3,4);

[camX, camY, camZ] = sphere(20);
radius = 0.015; % metri
surf(camX*radius + cam_origin(1), ...
     camY*radius + cam_origin(2), ...
     camZ*radius + cam_origin(3), ...
     'FaceColor','r','EdgeColor','none','FaceAlpha',0.7);

text(cam_origin(1)+0.02, cam_origin(2) -0.15, cam_origin(3) - 0.03, 'Camera', ...
    'Color','k','FontSize',10,'FontWeight','bold');

hold off;
