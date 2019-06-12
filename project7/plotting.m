%% Plotting for project 7
% Paige Lorson
% Parallel Programming

close all
clear
clc

%% Import data
addpath('C:\Users\Paige\Documents\MATLAB\myFunctions')
data{1} = importdata('p7OMP.txt',' ');
data{2} = importdata('p7SIMD.txt',' ');
data{3} = importdata('p7CUDA.txt',' ');

signal = [data{1}(1:511,2),data{1}(1:511,1),data{2}(1:511,1),data{3}(1:511,1)];
performance = [data{1}(512:end,:);data{2}(512:end,:);data{3}(512:end,:)];
performance = [performance(:,2),performance(:,1),[ones(8,1);2;3;3;3]];
p = [performance(1:8,1);0;0;performance(9,1);0;0;performance(10:12,1)];
load C

fig1 = figure(1); 
set(fig1,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
plot(signal(:,1),signal(:,2),'Color',c(1,:),'Marker','o','LineStyle','none','LineWidth',3)
plot(signal(:,1),signal(:,3),'Color',c(2,:),'Marker','x','LineStyle','none','LineWidth',2)
plot(signal(:,1),signal(:,4),'Color',c(3,:),'Marker','d','LineStyle','none','LineWidth',1)
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Shift','FontSize',18)
ylabel('Signal Sum','FontSize',18)
lgd1 = legend('OMP','SIMD','CUDA','Location','northwest'); 
title(lgd1,'Method','interpreter','latex','FontSize', 16)
title('Processed Signal','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')


fig2 = figure(2); 
set(fig2,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
% axes1 = axes('Parent',fig2);
% hold(axes1,'on');
b = bar(p);

TickLabel = {'1'; '2';'3';'4';...
             '5'; '6';'7';'8';...
             '16';'32';'64'};

set(gca,'XTick',[1 2 3 4 5 6 7 8 14 15 16],'XTickLabel',TickLabel)

set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel(strcat('                             OpenMP (Threads)',...
              '                                                  SIMD',...
              '                      CUDA (Block Size)'),...
        'FontSize',14)
ylabel('Mega Size per Second','FontSize',18)
% lgd1 = legend('16K','32K','64K','128K','256K','512K','Location','northwest'); 
% title(lgd1,'Trials','interpreter','latex','FontSize', 18)
title('Performance','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')

Speedup = p(:,1)/p(1,1);

fig3 = figure(3); 
set(fig3,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
b = bar(Speedup(2:end));

TickLabel = {'2';'3';'4';...
             '5'; '6';'7';'8';...
             '16';'32';'64'};

set(gca,'XTick',[1 2 3 4 5 6 7 13 14 15],'XTickLabel',TickLabel)

set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel(strcat('                        OpenMP (Threads)',...
              '                                                SIMD',...
              '                         CUDA (Block Size)'),...
        'FontSize',14)
ylabel('Speed up','FontSize',18)
title('Performance Increase','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')

% b.XTickLabelMode = 'manual';
% b.XTick = [1 2 3 4 5 6 7 8 12 16 17 18 19];
% b.XTickLabel = {'1 \n thread'; '2 \n threads';'3 \n threads';'4 \n threads';...
%                 '5 \n threads'; '6 \n threads';'7 \n threads';'8 \n threads';...
%                 'SIMD'; '16 \n Block Size';'32 \n Block Size';'64 \n Block Size'};


% bar([performance(8,2),0,0])
% bar([performance(7,2),0,performance(12,2)])
% bar([performance(6,2),0,performance(11,2)])
% bar([performance(5,2),performance(9,2),performance(10,2)])
