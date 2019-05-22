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

NUM_WORK_GROUPS{1} = unique(data{1}(:,3));
NUM_WORK_GROUPS{2} = unique(data{2}(:,3));
NUM_WORK_GROUPS{3} = unique(data{3}(:,3));

for i = 1:3
    L1 = length(NUM_ELEMENTS{i});
    for j = 1:L1
        L2 = length(LOCAL_SIZE{i});
        for k = 1:L2
            PERFORMANCE{i}(j,k) = data{i}(L2*(j-1)+k,4);
        end
    end
end

n1 = 3;
n2 = 4;
fig1 = figure(1); 
set(fig1,'units','normalized','outerposition',[0 0 0.75 1]);
for i = 1:2
    if i == 1
        LS = '--';
    end
    if i == 2
        LS = '-';
    end
    L2 = length(LOCAL_SIZE{i});

    for k = 1:n2:floor(L2/n2)*n2
        plot(NUM_ELEMENTS{i}(1:floor(L1/n2):end),PERFORMANCE{i}(1:floor(L1/n2):end,k),'Marker','o','LineStyle',LS,'LineWidth',3)
        hold on
    end
end

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
    for j = 1:n1:floor(L1/n1)*n1
        plot(LOCAL_SIZE{i}(1:floor(L2/n2):end),PERFORMANCE{i}(j,1:floor(L2/n2):end),'Marker','o','LineStyle',LS,'LineWidth',3)
        hold on
    end
end

fig3 = figure(3); 
set(fig3,'units','normalized','outerposition',[0 0 0.75 1]);

for i = 3
    if i == 1
        LS = '--';
    end
    if i == 2
        LS = '-';
    end
    L2 = length(LOCAL_SIZE{i});

    for k = 1:L2
        plot(NUM_ELEMENTS{i}(1:floor(L1/n2):end),PERFORMANCE{i}(1:floor(L1/n2):end,k),'Marker','o','LineStyle',LS,'LineWidth',3)
        hold on
    end
end

fig4 = figure(4); 
set(fig4,'units','normalized','outerposition',[0 0 0.75 1]);
for i = 3
    if i == 1
        LS = '--';
    end
    if i == 2
        LS = '-';
    end
    L1 = length(NUM_ELEMENTS{i});
    for j = 1:n1:floor(L1/n1)*n1
        plot(LOCAL_SIZE{i}(1:floor(L2/n2):end),PERFORMANCE{i}(j,1:floor(L2/n2):end),'Marker','o','LineStyle',LS,'LineWidth',3)
        hold on
    end
end
