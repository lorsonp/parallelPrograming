%% Plotting for project 2
% Paige Lorson
% Parallel Programming

close all
clear
clc

%% Import data
addpath('C:\Users\Paige\Documents\MATLAB\myFunctions')
data = importdata('project2.txt',' ');
save('data.mat','data')
NUMNODES = data(:,1);
NUMT = data(:,2);
maxPerformance = data(:,3);
volume = data(:,4);

%% Organization
maxNUMT = max(NUMT)-1;
variedNodes = length(NUMNODES)/(maxNUMT+1);


%% Present data

tabPerf = zeros(maxNUMT+1,variedNodes+1);
tabPerf(1,2:end) = NUMNODES(1:variedNodes);
tabPerf(:,1) = (0:maxNUMT)';
tabVol = tabPerf;
tabSpeedup = tabPerf;
tabFp = tabPerf;

L = 1;
for i = 2:maxNUMT+1 %loop through threads
    tabPerf(i,2:end) = maxPerformance(L:variedNodes-1+L)';
    tabVol(i,2:end) = volume(L:variedNodes-1+L)';
    L = L+variedNodes;
end
tabSpeedup(2:end,2:end) = tabPerf(2:end,2:end)./tabPerf(2,2:end);
tabFp(3:end,2:end) = ((2:maxNUMT)./(1:maxNUMT-1))'.*(1 - 1./tabSpeedup(3:end,2:end));
latex_tableA = latex(vpa(sym(tabPerf(:,1:11)),3));
latex_tableB = latex(vpa(sym(tabPerf(:,[1,12:end])),3));
disp(latex_tableA)
disp(latex_tableB)
avgFp = mean(mean(tabFp(7:end,5:end)));
maxSpeedUp = 1/(1-avgFp);

load color
skip = 1/2;
Lgd_nodes = cell(1,ceil(variedNodes*(1-skip)));
j=1;
for i = 1:1/skip:variedNodes
Lgd_nodes{j} = num2str(NUMNODES(i));
j=j+1;
end
Lgd_threads = {'1','2','3','4','5','6','7','8'};
% c = rand(20,3);
%%
fig1 = figure(1);    
set(fig1,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 2:2:variedNodes+1
plot(tabPerf(2:end,1),tabPerf(2:end,i),'Color',c(i,:),'LineStyle','-','LineWidth',3)
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Threads','FontSize',18)
ylabel('Mega Nodes per Second','FontSize',18)
lgd1 = legend(Lgd_nodes,'Location','northwest'); 
title(lgd1,'Nodes','interpreter','latex','FontSize', 18)
title('Mega Nodes per Second vs. Number of Threads','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')

fig2 = figure(2);    
set(fig2,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 2:maxNUMT+1
plot(tabPerf(1,2:end),tabPerf(i,2:end),'Color',c(i,:),'LineStyle','-','LineWidth',3)
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman','XScale','linear') 
xlabel('Nodes','FontSize',18)
ylabel('Mega Nodes per Second','FontSize',18)
lgd1 = legend(Lgd_threads{1:end-1},'Location','northwest'); 
title(lgd1,'Threads','interpreter','latex','FontSize', 18)
title('Mega Nodes per Second vs. Number of Nodes','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')
%%
fig3 = figure(3);    
set(fig3,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 2:maxNUMT+1
plot(tabVol(1,2:end),tabVol(i,2:end),'Color',c(i,:),'LineStyle','-','LineWidth',3)
end
Vol_avg = mean(tabVol(2:end,end));
plot([tabVol(1,2),tabVol(1,end)],[Vol_avg,Vol_avg],'--k','LineWidth',4)
% disp(tabVol(i,end))
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman','XScale','linear') 
xlabel('Nodes','FontSize',18)
ylabel('Volume','FontSize',18)
lgd1 = legend(Lgd_threads{1:end-1},'Location','northwest'); 
title(lgd1,'Threads','interpreter','latex','FontSize', 18)
title('Volume vs. Number of Nodes','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')

%%
fig4 = figure(4);
set(fig4,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 2:2:variedNodes+1
plot(tabSpeedup(2:end,1),tabSpeedup(2:end,i),'Color',c(i,:),'LineStyle','-','LineWidth',3)
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Threads','FontSize',18)
ylabel('Speedup','FontSize',18)
lgd1 = legend(Lgd_nodes,'Location','northwest'); 
title(lgd1,'Nodes','interpreter','latex','FontSize', 18)
title('Speedup vs Number of Threads','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')

fig5 = figure(5);
set(fig5,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 3:maxNUMT+1
plot(tabSpeedup(1,2:end),tabSpeedup(i,2:end),'Color',c(i,:),'LineStyle','-','LineWidth',3)
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman','XScale','linear') 
xlabel('Nodes','FontSize',18)
ylabel('Speedup','FontSize',18)
lgd1 = legend(Lgd_threads{2:end-1},'Location','northwest'); 
title(lgd1,'Threads','interpreter','latex','FontSize', 18)
title('Speedup vs Number of Nodes','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')

%% 
fig6 = figure(6);
set(fig6,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 2:2:variedNodes+1
plot(tabFp(3:end,1),tabFp(3:end,i),'Color',c(i,:),'LineStyle','-','LineWidth',3)
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Threads','FontSize',18)
ylabel('F_p','FontSize',18)
lgd1 = legend(Lgd_nodes,'Location','northwest'); 
title(lgd1,'Nodes','interpreter','latex','FontSize', 18)
title('Parallel Fraction vs Number of Threads','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')

