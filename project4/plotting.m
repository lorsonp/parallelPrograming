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
         
% ArraySize = [ArraySize,ArraySize,ArraySize,ArraySize]; 
data1 =  (data1(1:17) + data1(18:34) + data1(35:51)...
       + data1(52:68) + data1(69:85) + data1(86:end))...
       / 6;
data2 =  (data2(1:17) + data2(18:34) + data2(35:51)...
       + data2(52:68) + data2(69:85) + data2(86:end))...
       / 6;

data3 =  (data3(1:17) + data3(18:34) + data3(35:51)...
       + data3(52:68) + data3(69:85) + data3(86:end))...
       / 6;
data4 =  (data4(1:17) + data4(18:34) + data4(35:51)...
       + data4(52:68) + data4(69:85) + data4(86:end))...
       / 6;
S_sumMult = data2./data4;
S_Mult = data1./data3;
table = [data1,data2,data3,data4];
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
semilogx(ArraySize,data1,'LineStyle','-','LineWidth',3)
hold on
semilogx(ArraySize,data2,'LineStyle','-','LineWidth',3)
semilogx(ArraySize,data3,'LineStyle','-','LineWidth',3)
semilogx(ArraySize,data4,'LineStyle','-','LineWidth',3)

set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Array Size','FontSize',18)
ylabel('Performance','FontSize',18)

legend('simd.Mult','simd.sumMult','nonsimd.Mult','nonsimd.sumMult','Location','south','Orientation','horizontal','FontSize',14); 
title('Results for Simulation','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')
