data = [0 0.5;0.8 1;1.6 4;3 5;4.0 6;5.0 8];%C1: X, C2: Y

%grouping data as X and Y
x = data(:,1);
x = [ones(size(x,1),size(x,2)) x];
y = data(:,2) ;

%standart LLS, w = (X^TX)^-1X^Td
w = inv(x.'*x)*x.'*y;
fit_x = -1:0.001:6;
fit_y = fit_x*w(2)+w(1);
%display the LLS fit
figure(1)
plot(fit_x,fit_y,x(:,2),y,'o')
xlabel("x")
ylabel("y")
title("LLS Algorithm"+"y="+w(2)+"x+"+w(1))
saveas(gcf,'LLS.png')

%LMS using gradient descent method
%e(n) = d(n)-w^T(n)x(n)
%w(n+1) = w(n)+learningParameter e(n)x(n)
%Learning Rate 0.01
learning = 0.01; %learning rate
wb = [0.2;1]; %randomly selected weights
epochh = 1:100;
avg_e = zeros(size(epochh,2),1); %array to store avg error at each epoch 
for epoch = epochh
    for i = 1:size(x,1)
        e = y(i)-wb.'*x(i,:).';  %error calculation
        fprintf("Epoch %d error %i is %0.4f\n",epoch,i,e);
        avg_e(epoch) =avg_e(epoch) +e^2/2;
        wb = wb+learning*e*x(i,:).'; %weight update based on gradient descent 
    end
    avg_e(epoch) =avg_e(epoch)/4;
end
%display the LMS fit
figure(2)
fitB_x = -1:0.001:6;
fitB_y = fitB_x*wb(2)+wb(1) ;
plot(fitB_x,fitB_y,x(:,2),y,'o')
xlabel("x")
ylabel("y")
title("LMS Algorithm with learning rate= "+learning+" y="+wb(2)+"x+"+wb(1))
saveas(gcf,'LMS.png')

%display the error vs epoch
figure(3)
plot(epochh,avg_e)
title("Average error e^2/2 at the end of each epoch, LR ="+learning)
ylabel("Error 1/2e^2");
xlabel("Epoch")
saveas(gcf,'epoch.png')

%display the LLS-LMS to see the order of difference between each method.
figure(4)
dif_y = fitB_y-fit_y;
dif_w1 = wb(2)-w(2);
dif_b = wb(1) - w(1);
plot(fit_x,dif_y)
title("LMS-LLS for learning rate= "+learning+" y="+dif_w1+"x+"+dif_b)
saveas(gcf,'Difference.png')