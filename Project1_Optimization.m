
function w0 = Project1_Optimization(data, regularize)
    data_shuffled = data(randperm(size(data,1)),:);
    [rows,cols] = size(data);
    for round = 1:5
        test_start = uint32(0.2*(round-1)*rows+1);%the start data point of test data in the entire dataset
        if round ~= 5
            test_end = uint32(test_start + 0.2*rows);%the last data point of test data in the entire dataset
        else
            test_end = rows;
        end 
        test_data = data_shuffled(test_start:test_end,:);%select the testing data
        if round == 1
            train_data = data_shuffled(test_end+1:end,:);
        elseif round == 5
            train_data = data_shuffled(1:test_start-1,:);
        else 
            train_data = cat(1,data_shuffled(1:test_start-1,:),data_shuffled(test_end+1:end,:));
        end 
        train_pars = train_data(:,end);
        test_pars = test_data(:,end);
        train_data = im2double(train_data);%==========================================>
        test_data = im2double(test_data);%==========================================>
        w0 = optimization(train_data,test_data,train_pars, test_pars,regularize,round);
    end 
end

function w0 = optimization(train_data, test_data, train_pars,test_pars, regularize,num_round)  
  
   fprintf(['Training \n']);
   k=10;
   lambda = 0.1;%L2-regularization coefficient
   count_error = 1;%count the number of errors computed
   
   %set up all data
   train_data = train_data(:,1:end-1);%remove the last column since it is label column
   test_data = test_data(:,1:end-1);    
   [row_train,n]=size(train_data);% row_train is the row number of data in train_data, n is the number of features
   w0 = zeros(n+1,k); %initial weights, n is the number of features of each data point, k is the number of classes    
   last_column = ones(row_train,1);
   train_data = [train_data,last_column];%add one column of ones to the end of train_data so that the dimension matches with w0
   [row_test,~] = size(test_data);
   last_column = ones(row_test,1);
   test_data = [test_data,last_column];%add one column of ones to the end of test_data so that the dimension matches with w0  
   max_train_data = max(train_data);%get the max values of each column of train data
   train_data = bsxfun(@rdivide,train_data,max_train_data);
   max_test_data = max(test_data);%get the max values of each column of train data
   test_data = bsxfun(@rdivide,test_data,max_test_data);
   %end set up all data
   
  % The function handle of the objective and gradient
  obj = @(w0)my_obj(w0, train_data,train_pars);
  
  % function handles which evaluate the train and test error at each iteration
  for i = 1:2
    if i == 1
        eval_error{i} = @(w0, optimValues, state)my_test(w0, train_data,train_pars);%set the i-th element to the rvalue
    else
        eval_error{i} = @(w0, optimValues, state)my_test(w0, test_data,test_pars);
    end
  end

  opt = optimset('display', 'iter-detailed', ... % print detailed information at each iteration of optimization
                 'LargeScale', 'off', ... % This makes sure that quasi-Newton algorithm is used. Do not use the active set algorithm (when LargeScale is set to 'on')
                 'GradObj', 'on', ... % the function handle supplied in calling fminunc provides gradient information
                 'MaxIter', 400, ...  % Run maximum 100 iterations. Terminate after that.
                 'MaxFunEvals', 100, ...  % Allow CRF objective/gradient to be evaluated at most 100 times. Terminate after that.
                 'TolFun', 1e-6, ...  % Terminate when tolerance falls below 1e-3
                 'OutputFcn', eval_error);  % each iteration, invoke the function handles to print the training and test error of the current model
                                            % note this could be very expensive, so only for verification purpose
  
  [w0, f, flag] = fminunc(obj, w0, opt);%flag is the exit condition of fminunc, fval is the objective value, opt is the structure output
 
  
  [~, accuracy] = my_test(w0, test_data,test_pars);
  fprintf('my test accuracy: %g\n', accuracy);
  
  % Compute the objective and gradient on the data
  % evaluated at the current model, stored as a vector)
  function [f, g] = my_obj(x, data,train_pars)
        
        [num_train,~]=size(data);% num_train is the number of data in train_data
        label = zeros(num_train,k);%used to convert the pars vector to a matrix. 1-->(0,1,0,...,0)
        train_pars = uint8(train_pars);%convert the labels to integer type
        for m = 1:num_train
            if train_pars(m)==0 || train_pars(m)==1 || train_pars(m)==2 || train_pars(m)==3|| train_pars(m)==4|| train_pars(m)==5|| train_pars(m)==6|| train_pars(m)==7|| train_pars(m)==8|| train_pars(m)==9
                label(m,train_pars(m)+1)=1;
            end 
        end 
      
        f = get_my_obj(data, x, label); % compute the objective value
        total_error(count_error) = f;%store error of each epoch into total_error
        count_error = count_error + 1;
        
    %    disp('weight');
     %   disp(x);
        
        g = get_my_grad(data, x, label); % compute the gradient
  end
%PLOT the errors
%*******************************************************************************************
  subplot(3,2,num_round);
  plot(total_error);% plot the errors and epochs.
  title(strcat('Error value - ',num2str(num_round)));

%The following are my functions.--------------------------------*
    function f = get_my_obj(data,x,pars)   
       
       	 numerator_ = exp(-(double(data)*double(x)));
         denominator_ = sum(numerator_,2);%compute the sum vector of each row
         y_ = bsxfun(@rdivide,numerator_,denominator_);
         labeled_y_ = log(y_).*pars;
         if regularize==true
             f = -sum(sum(labeled_y_,2))+2*lambda*norm(x)*norm(x);% .* element wise multiplication
         else
             f = -sum(sum(labeled_y_,2));
         end
    end

    function g = get_my_grad(data,x,pars)
         numerator_ = exp(-(double(data)*double(x)));
         denominator_ = sum(numerator_,2);%compute the sum vector of each row
         y_ = bsxfun(@rdivide,numerator_,denominator_);
         [d_features,n_classes] = size(x);%get the number of features with bias and the number of classes
         y_t = y_-pars;       
         [n_data,~] = size(data);
         derivative = zeros(n_data,n_classes,d_features);
         derivative_error = zeros(n_classes,d_features);
         for p = 1:n_classes
             for q = 1:n_data
                 for r = 1:d_features
                     derivative(q,p,r) = y_t(q,p)*data(q,r);
                 end 
             end     
         end    
        for t = 1:n_classes
            derivative_error(t,:) = sum(derivative(:,t,:));
        end  
        if regularize==true
            derivative_error = transpose(derivative_error) + 2*lambda*x;%regularization
        else 
            derivative_error = transpose(derivative_error);
        end 
        
        g = -derivative_error;
    end
%up to this line.-----------------------------------------------*

  % Compute the test accuracy on the test data
  % w is the current model
  function [stop, accuracy] = my_test(w, data,pars)
    stop = false;   % solver can be terminated if stop is set to true
    y_predict = my_predict(w, data);
    
    % Compute test accuracy by comparing the prediction with the ground truth
    accuracy = compare(y_predict, pars);
  end

%The following are my functions.--------------------------------*
    function y_predict = my_predict(w,data)
       
         numerator_ = exp(-(double(data)*double(w)));
         denominator_ = sum(numerator_,2);%compute the sum vector of each row
         matrix_predict = bsxfun(@rdivide,numerator_,denominator_);
         [~,max_index] = max(transpose(matrix_predict));
         y_predict = transpose(max_index)-1;
       
    end

    function accuracy = compare(y_predict, groundtruth)
        [len,~] = size(y_predict);
        count = 0;
        for s = 1:len
            if(y_predict(s)==groundtruth(s))
                count = count + 1;
            end 
        end 
        accuracy = count/len;
    end

%up to this line.-----------------------------------------------*
end
