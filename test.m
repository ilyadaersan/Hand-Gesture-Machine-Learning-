
close all
clc


fs = 2000; % sampling rate "Hz"

%% Get sEMG Records directory:
current_folder= 'C:\Users\ASUS\Desktop\MachineLearning'; %  !!!change with current folder!!!
addpath(genpath(current_folder))
Base  = strcat(current_folder,'\test'); % train or test
List  = dir(fullfile(Base, '**', '*.csv'));
Files = fullfile({List.folder}, {List.name});
Nd =  cellfun( @str2double, regexp(Files, '\d+', 'match') );
[~,I] = sort(Nd);
Files = Files(I);

D = {};

%%Using for automated segmentation of all participant sEMG data to sEMG gesture segment.
for iFile = 1:numel(Files) %40 participants
    
    data = load(Files{iFile}); % load .. sEMG data of participants
   

        for gesture = 0:9 % a total of 10 hand gesture
            
            if gesture==0
                 data1 =data(1:5,:); 
                 gesture_label='Res';
            elseif gesture==1
                 data1=data(6:10,:); 
                 gesture_label='Ext';
            elseif gesture==2
                 data1=data(11:15,:); 
                 gesture_label='Fle';
            elseif gesture==3
                 data1=data(16:20,:); 
                 gesture_label='Uln';
            elseif gesture==4
                 data1=data(21:25,:); 
                 gesture_label='Rad';
            elseif gesture==5
                 data1=data(26:30,:); 
                 gesture_label='Gri';
            elseif gesture==6
                 data1=data(31:35,:); 
                 gesture_label='Abd';
            elseif gesture==7
                 data1=data(36:40,:); 
                 gesture_label='Add';
            elseif gesture==8
                 data1=data(41:45,:); 
                 gesture_label='Sup';
            elseif gesture==9
                 data1=data(46:50,:); 
                 gesture_label='Pro';
            end % 
            
            %0: X = REST
            %1: E = EXTENSION
            %2: F = FLEXION
            %3: U = ULNAR DEVIATION
            %4: R = RADIAL DEVIATION
            %5: G = GRIP
            %6: B = ABDUCTION
            %7: D = ADDUCTION
            %8: S = SUPINATION
            %9: P = PRONATION
      
      for rep = 1:5 % 5 repetition
        
           data2 = data1(rep,:);
      
         
           for i =  0: 100: 7500 % sliding window approach
                ch1 = data2(1+i:500+i);
                ch2 = data2(8001+i : 8500+i);
                ch3 = data2(16001+i :16500+i);
                ch4 = data2(24001+i : 24500+i);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%                 USE HERE "ch1 ch2 ch3 ch4" TO ANALYZE                       %%%                                                                      %%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                MAV1 = mean(abs(ch1));
                MAV2 = mean(abs(ch2));
                MAV3 = mean(abs(ch3));
                MAV4 = mean(abs(ch4));
                
                D1 = [];
                d1 = [D1 MAV1,MAV2,MAV3,MAV4];
                
                STD1 = std(ch1);
                STD2 = std(ch2);
                STD3 = std(ch3);
                STD4 = std(ch4);
                
                D2 = [];
                d2 = [D2 STD1,STD2,STD3,STD4];
                
                VAR1 = var(ch1);
                VAR2 = var(ch2);
                VAR3 = var(ch3);
                VAR4 = var(ch4);
                D3 = [];
                d3 = [D3 VAR1,VAR2,VAR3,VAR4];
                
                K   = length(ch1); 
                MSR1 = (1 / K) * sum(ch1 .^ (1/2));
                L   = length(ch2); 
                MSR2 = (1 / L) * sum(ch2 .^ (1/2));
                M   = length(ch3); 
                MSR3 = (1 / M) * sum(ch3 .^ (1/2));
                N   = length(ch4); 
                MSR4 = (1 / N) * sum(ch4 .^ (1/2));
                
                D4 =[];
                d4 = [D4 MSR1,MSR2,MSR3,MSR4];

                
                
              
                
                
                
                B = {d1,d2,d3,d4};
                D{end+1}= cat(2,B);

           end% sliding window for
    
         
       end% repetition for
     
    end%gesture for

end%participants for

D=D.';
D=vertcat(D{:});
D=cell2mat(D);
