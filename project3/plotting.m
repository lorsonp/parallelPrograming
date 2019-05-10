%% Plotting for project 3
% Paige Lorson
% Parallel Programming

close all
clear
clc

%% Import data
addpath('C:\Users\Paige\Documents\MATLAB\myFunctions')
data = importdata('project3.txt',' ');
save('data.mat','data')
year = data(:,1);
month = data(:,2);
temp = data(:,3);
precip = data(:,4);
height = data(:,5);
deer = data(:,6);
income = data(:,7);
temp_plot = (5./9.)*(temp-32);
year_plot = year + (month+1)/12;
data(:,2) = data(:,2) + 1;
latex_table = latex(vpa(sym(data),3));

c = rand(4,3);
%% 
fig1 = figure(1); 

set(fig1,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
yyaxis left
% plot(year_plot,height,'LineStyle','-','LineWidth',3,'Color',c(1,:))
% plot(year_plot,deer,'LineStyle','-','LineWidth',3,'Color',c(2,:))
% plot(year_plot,temp,'LineStyle','-','LineWidth',3,'Color',c(3,:))
% plot(year_plot,precip,'LineStyle','-','LineWidth',3,'Color',c(4,:))
plot(year_plot,height,'LineWidth',2)
plot(year_plot,deer,'LineWidth',2)
plot(year_plot,temp,'LineWidth',2)
plot(year_plot,precip,'LineWidth',2)
ylim([-5,inf])
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Years','FontSize',18)
ylabel('Results','FontSize',18)

yyaxis right
plot(year_plot,income,'LineWidth',2)
ylim([-50,inf])
set(gca,'LineWidth',0.5,'FontName','Times New Roman') 
xlabel('Years','FontSize',18)
ylabel('Income ($)','FontSize',18)
legend('Height of grain (in)','Number of Deer','Tempurature (°C)',...
    'Precipitation','Income from Harvest ($)','Location','south','Orientation','horizontal','FontSize',14); 
title('Results for Simulation','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')
