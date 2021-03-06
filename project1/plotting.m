Dr. Bailley,  
For project 2 I am seeing very little effect of number of nodes on the performance results. I am currently varying the number of nodes from 50 to 1000. In this a large enough range or should I include more be able to see the effect of overhead?





%% Plotting for project 1
% Paige Lorson
% Parallel Programming

close all
clear
clc

%% Import data
addpath('C:\Users\Paige\Documents\MATLAB\myFunctions')
data = importdata('project1.txt',' ');
save('data.mat','data')
NUMTRIALS = data(:,1);
NUMT = data(:,2);
maxPerformance = data(:,3);
currentProb = data(:,4);

%% Present data

tabPerf = zeros(9,6);
tabPerf(1,2:end) = NUMTRIALS(1:5);
tabPerf(:,1) = (0:8)';
tabProb = tabPerf;
tabSpeedup = tabPerf;
tabFp = tabPerf;
L = 1;
for i = 2:9 %loop through threads
    tabPerf(i,2:end) = maxPerformance(L:4+L)';
    tabProb(i,2:end) = currentProb(L:4+L)';
    L = L+5;
end
tabSpeedup(2:end,2:end) = tabPerf(2:end,2:end)./tabPerf(2,2:end);
tabFp(3:end,2:end) = (2:8./(1:7))'.*(1 - 1./tabSpeedup(3:end,2:end));
latex_table = latex(vpa(sym(tabPerf),3));
disp(latex_table)

load color
% c = rand(10,3);
%%
fig1 = figure(1);    
set(fig1,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 2:6
plot(tabPerf(2:end,1),tabPerf(2:end,i),'Color',c(i,:),'LineStyle','-','LineWidth',3)
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Threads','FontSize',18)
ylabel('Mega Trials per Second','FontSize',18)
lgd1 = legend('100','1,000','10,000','100,000','1,000,000','Location','northwest'); 
title(lgd1,'Trials','interpreter','latex','FontSize', 18)
title('Mega Trials per Second vs. Number of Threads','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')

fig2 = figure(2);    
set(fig2,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 2:9
plot(tabPerf(1,2:end),tabPerf(i,2:end),'Color',c(i,:),'LineStyle','-','LineWidth',3)
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman','XScale','log') 
xlabel('Trials','FontSize',18)
ylabel('Mega Trials per Second','FontSize',18)
lgd1 = legend('1','2','3','4','5','6','7','8','Location','northwest'); 
title(lgd1,'Threads','interpreter','latex','FontSize', 18)
title('Mega Trials per Second vs. Number of Trials','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')
%%
fig3 = figure(3);    
set(fig3,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 2:9
plot(tabProb(1,2:end),tabProb(i,2:end),'Color',c(i,:),'LineStyle','-','LineWidth',3)
end
plot([tabProb(1,2),tabProb(1,end)],[tabProb(i,end),tabProb(i,end)],'--k','LineWidth',4)
disp(tabProb(i,end))
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman','XScale','log') 
xlabel('Trials','FontSize',18)
ylabel('Probability','FontSize',18)
lgd1 = legend('1','2','3','4','5','6','7','8','Location','northwest'); 
title(lgd1,'Threads','interpreter','latex','FontSize', 18)
title('Probability vs. Number of Trials','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')

%%
fig4 = figure(4);
set(fig4,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 2:6
plot(tabSpeedup(2:end,1),tabSpeedup(2:end,i),'Color',c(i,:),'LineStyle','-','LineWidth',3)
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Threads','FontSize',18)
ylabel('Speedup','FontSize',18)
lgd1 = legend('100','1,000','10,000','100,000','1,000,000','Location','northwest'); 
title(lgd1,'Trials','interpreter','latex','FontSize', 18)
title('Speedup vs Number of Threads','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')

fig5 = figure(5);
set(fig5,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 3:9
plot(tabSpeedup(1,2:end),tabSpeedup(i,2:end),'Color',c(i,:),'LineStyle','-','LineWidth',3)
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman','XScale','log') 
xlabel('Trials','FontSize',18)
ylabel('Speedup','FontSize',18)
lgd1 = legend('2','3','4','5','6','7','8','Location','northwest'); 
title(lgd1,'Threads','interpreter','latex','FontSize', 18)
title('Speedup vs Number of Trials','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')

%% 
fig6 = figure(6);
set(fig6,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 2:6
plot(tabFp(3:end,1),tabFp(3:end,i),'Color',c(i,:),'LineStyle','-','LineWidth',3)
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Threads','FontSize',18)
ylabel('F_p','FontSize',18)
lgd1 = legend('100','1,000','10,000','100,000','1,000,000','Location','northwest'); 
title(lgd1,'Trials','interpreter','latex','FontSize', 18)
title('Parallel Fraction vs Number of Threads','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')

