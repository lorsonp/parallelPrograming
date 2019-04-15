%% Plotting for project 1
% Paige Lorson
% Parallel Programming

close all
clear
clc

%% Import data

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



fig1 = figure(1);    
set(fig1,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 2:6
plot(tabPerf(2:end,1),tabPerf(2:end,i),'LineStyle','-','LineWidth',3)
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Threads','FontSize',18)
ylabel('Mega Trials per Second','FontSize',18)
lgd1 = legend('100','1,000','10,000','100,000','1,000,000','Location','northwest'); 
title(lgd1,'Trials','interpreter','latex','FontSize', 18)
title('Total Runtime','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')



fig2 = figure(2);    
set(fig2,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 2:9
plot(tabPerf(1,2:end),tabPerf(i,2:end),'LineStyle','-','LineWidth',3)
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman','XScale','log') 
xlabel('Trials','FontSize',18)
ylabel('Mega Trials per Second','FontSize',18)
lgd1 = legend('1','2','3','4','5','6','7','8','Location','northwest'); 
title(lgd1,'Threads','interpreter','latex','FontSize', 18)
title('Total Runtime','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')

%% 
fig3 = figure(3);    
set(fig3,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 2:9
plot(tabProb(1,2:end),tabProb(i,2:end),'LineStyle','-','LineWidth',3)
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman','XScale','log') 
xlabel('Trials','FontSize',18)
ylabel('Mega Trials per Second','FontSize',18)
lgd1 = legend('1','2','3','4','5','6','7','8','Location','northwest'); 
title(lgd1,'Threads','interpreter','latex','FontSize', 18)
title('Total Runtime','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')


fig4 = figure(4);    
set(fig4,'units','normalized','outerposition',[0 0 0.75 1]);
hold on
for i = 2:6
plot(tabSpeedup(2:end,1),tabSpeedup(2:end,i),'LineStyle','-','LineWidth',3)
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Threads','FontSize',18)
ylabel('Mega Trials per Second','FontSize',18)
lgd1 = legend('100','1,000','10,000','100,000','1,000,000','Location','northwest'); 
title(lgd1,'Trials','interpreter','latex','FontSize', 18)
title('Total Runtime','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')
