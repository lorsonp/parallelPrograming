%% Plotting for project 3
% Paige Lorson
% Parallel Programming

close all
clear
clc

%% Import data
addpath('C:\Users\Paige\Documents\MATLAB\myFunctions')
data1 = importdata('project4.simd.Mult.txt',' ');
data2 = importdata('project4.simd.sumMult.txt',' ');
data3 = importdata('project4.nonsimd.Mult.txt',' ');
data4 = importdata('project4.nonsimd.sumMult.txt',' ');
ArraySize = [1000,2500,5000,7500,10000,25000,50000,75000,...
             100000,250000,500000,750000,1000000,2500000,...
             5000000,7500000,10000000];
S_sumMult = data4./data2;
S_Mult = data3./data1;
table = [ArraySize',ArraySize'./[data1,data2,data3,data4]/1000];
latex(vpa(sym(table),4))

fig1 = figure(1); 
set(fig1,'units','normalized','outerposition',[0 0 0.75 1]);
semilogx(ArraySize,S_sumMult,'LineStyle','-','LineWidth',3)
hold on
semilogx(ArraySize,S_Mult,'LineStyle','-','LineWidth',3)

set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Array Size','FontSize',18)
ylabel('Speed-up','FontSize',18)

legend('Sum-Mult','Mult','Location','south','Orientation','horizontal','FontSize',14); 
title('Results for Simulation','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')



fig2 = figure(2); 
set(fig2,'units','normalized','outerposition',[0 0 0.75 1]);
semilogx(ArraySize,ArraySize'./data1/1000,'LineStyle','-','LineWidth',3)
hold on
semilogx(ArraySize,ArraySize'./data2/1000,'LineStyle','-','LineWidth',3)
semilogx(ArraySize,ArraySize'./data3/1000,'LineStyle','-','LineWidth',3)
semilogx(ArraySize,ArraySize'./data4/1000,'LineStyle','-','LineWidth',3)

set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Array Size','FontSize',18)
ylabel('Performance','FontSize',18)

legend('simd.Mult','simd.sumMult','nonsimd.Mult','nonsimd.sumMult','Location','south','Orientation','horizontal','FontSize',14); 
title('Results for Simulation','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')


