%% Plotting for project 5
% Paige Lorson
% Parallel Programming

close all
clear
clc

%% Import data
addpath('C:\Users\Paige\Documents\MATLAB\myFunctions')
data{1} = importdata('part1.txt',' ');
data{2} = importdata('part2.txt',' ');
data{3} = importdata('part3.txt',' ');

NUM_ELEMENTS{1} = unique(data{1}(:,1));
NUM_ELEMENTS{2} = unique(data{2}(:,1));
NUM_ELEMENTS{3} = unique(data{3}(:,1));

LOCAL_SIZE{1} = unique(data{1}(:,2));
LOCAL_SIZE{2} = unique(data{2}(:,2));
LOCAL_SIZE{3} = unique(data{3}(:,2));


for i = 1:3
    L1 = length(NUM_ELEMENTS{i});
    for j = 1:L1
        L2 = length(LOCAL_SIZE{i});
        for k = 1:L2
            PERFORMANCE{i}(j,k) = data{i}(L2*(j-1)+k,4);
        end
    end
    tab = [[0,LOCAL_SIZE{i}'];[NUM_ELEMENTS{i},PERFORMANCE{i}]];
    latex_table = latex(vpa(sym(tab),4));
    if i == 3
        latex_table = latex(vpa(sym(tab),6));
    end
disp(latex_table)
end


n1 = 3;
n2 = 4;
load C
% c = rand(30,3);
fig1 = figure(1); 
set(fig1,'units','normalized','outerposition',[0 0 0.75 1]);

for i = 1:2
    if i == 1
        LS = '-';
    end
    if i == 2
        LS = '--';
    end
    L2 = length(LOCAL_SIZE{i});

    for k = 1:L2
        plot(NUM_ELEMENTS{i}(1:floor(L1/n2):end),PERFORMANCE{i}(1:floor(L1/n2):end,k),'Color',c(k,:),'Marker','o','LineStyle',LS,'LineWidth',3)
        hold on
    end
end
set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Number of Elements','FontSize',18)
ylabel('Giga-Calcs per Second','FontSize',18)
lgd1 = legend('32','64','128','256','512','1024',...
              'Location','northwest','NumColumns',L2); 
title(lgd1,'Local Size ( -- Mult, - - MultAdd )',...
      'interpreter','latex','FontSize', 16)
title('Mult and Mult-Add','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')


fig2 = figure(2); 
set(fig2,'units','normalized','outerposition',[0 0 0.75 1]);
for i = 1:2
    if i == 1
        LS = '--';
    end
    if i == 2
        LS = '-';
    end
    L1 = length(NUM_ELEMENTS{i});
    for j = 1:L1
        plot(LOCAL_SIZE{i}(1:floor(L2/n2):end),PERFORMANCE{i}(j,1:floor(L2/n2):end),'Color',c(j,:),'Marker','o','LineStyle',LS,'LineWidth',3)
        hold on
    end
end

set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Local Size','FontSize',18)
ylabel('Giga-Calcs per Second','FontSize',18)
lgd1 = legend('1024','10240','102400','1024000','10240000',...
              'Location','northwest','NumColumns',L2); 
title(lgd1,'Number of Elements ( -- Mult, - - MultAdd )','interpreter','latex','FontSize', 16)
title('Mult and Mult-Add','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')

fig3 = figure(3); 
set(fig3,'units','normalized','outerposition',[0 0 0.75 1]);

for i = 3
    LS = '-';
    L2 = length(LOCAL_SIZE{i});

    for k = 1:L2
        plot(NUM_ELEMENTS{i}(1:floor(L1/n2):end),PERFORMANCE{i}(1:floor(L1/n2):end,k),'Color',c(k,:),'Marker','o','LineStyle',LS,'LineWidth',3)
        hold on
    end
end

set(gca,'LineWidth',0.5,'FontSize',12,'FontName','Times New Roman') 
xlabel('Number of Elements','FontSize',18)
ylabel('Giga-Calcs per Second','FontSize',18)
lgd1 = legend('32','64','128','256','512','1024',...
              'Location','northwest','NumColumns',L2); 
title(lgd1,'Local Size','interpreter','latex','FontSize', 16)
title('Mult-Reduction','interpreter','latex','FontSize', 20)
set(gcf,'Color','w')
