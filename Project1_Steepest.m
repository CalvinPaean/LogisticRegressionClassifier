
function w0 = Project1_Steepest(data, regularize)
%    data = im2double(data);
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
        w0 = steepest(train_data,test_data,train_pars, test_pars,regularize,round);
    end  
end 
function w0 = steepest(train_data, test_data, pars_train, pars_test, regularize,num_round)
    
    fprintf('Training \n');
    stepsize = 0.00001; %learning rate
   % eps = 1e-5; %threshold of gradient  
    lambda = 0.1;%L2-regularization coefficient
    k = 10;% the number of classes 
    
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
    label = zeros(row_train,k);%used to convert the pars vector to a matrix. 1-->(0,1,0,...,0)
    pars_train = uint32(pars_train);%convert the labels to integer type
    for m = 1:row_train
        if pars_train(m)==0 || pars_train(m)==1 || pars_train(m)==2 || pars_train(m)==3|| pars_train(m)==4|| pars_train(m)==5|| pars_train(m)==6|| pars_train(m)==7|| pars_train(m)==8|| pars_train(m)==9
            label(m,pars_train(m)+1)=1;
        end 
    end 
   
    [error,derivative_error] = compute_error(w0,train_data,label);
    total_error(1) = error;%total error stores all errors in each computation
    
     w0 = w0 + stepsize*derivative_error;
    [error2,derivative_error] = compute_error(w0,train_data,label);
    total_error(2) = error2;
    count_error = 3;
    %disp(checkgrad('compute_error',w0,train_data,label,eps));%check the gradient computed correct or not
   
    while   error2<error %&& norm(derivative_error)>1   
        
        if error2<error
            stepsize = 1.01*stepsize; 
        else
            stepsize = stepsize*0.5;
        end
        w0 = w0 + stepsize*derivative_error;     
        error = error2;
        [error2,derivative_error] = compute_error(w0,train_data,label);  
        total_error(count_error) = error2;
        count_error = count_error+1;
    end
    subplot(3,2,num_round);
   
    plot(total_error);% plot the errors and epochs.
    title(strcat('Error value - ',num2str(num_round)));
   
    accuracy = my_test(w0, test_data, pars_test);
    disp('Accuracy: ');
    disp(accuracy);%print the accuracy
 
    function [error,derivative_error] = compute_error(w0, train_data, pars)   
         numerator_ = exp(-(train_data*w0));
         denominator_ = sum(numerator_,2);%compute the sum vector of each row
         y_ = bsxfun(@rdivide,numerator_,denominator_);
         labeled_y_ = log(y_).*pars;
         if regularize==true
              error = -sum(sum(labeled_y_,2))+2*lambda*norm(w0)*norm(w0);% .* element wise multiplication
         else
             error = -sum(sum(labeled_y_,2));
         end 
        
         [d_features,n_classes] = size(w0);%get the number of features with bias and the number of classes
         y_t = y_-pars;       
         [n_data,~] = size(train_data);
         derivative = zeros(n_data,n_classes,d_features);
         derivative_error = zeros(n_classes,d_features);
     
         for p = 1:n_classes
             for q = 1:n_data
                 for r = 1:d_features
                     derivative(q,p,r) = y_t(q,p)*train_data(q,r);
                 end 
             end     
         end    
        for t = 1:n_classes
            derivative_error(t,:) = sum(derivative(:,t,:));
        end  
        if regularize == true
             derivative_error = transpose(derivative_error) + 2*lambda*w0;%regularization
        else
            derivative_error = transpose(derivative_error);
        end 
    end%compute_error

    function accuracy = my_test(w, data,pars_test)
        y_predict = my_predict(w, data);
        accuracy = compare(y_predict, pars_test);
    end

    function y_predict = my_predict(w,data) 
         numerator_ = exp(-(data*w));
         denominator_ = sum(numerator_,2);%compute the sum vector of each row
         matrix_predict = bsxfun(@rdivide,numerator_,denominator_);   
       
         [~,max_index] = max(transpose(matrix_predict));
         y_predict = transpose(max_index)-1;
    end

    function accuracy = compare(y_predict, groundtruth)
   %     disp('Predicted classes£º');
    %   disp(y_predict);
       
        [len,~] = size(y_predict);
        count = 0;
        for s = 1:len
            if(y_predict(s)==groundtruth(s))
                count = count + 1;
            end 
        end 
        accuracy = count/len;
    end
end
