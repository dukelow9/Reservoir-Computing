%% ==================================== FIGURES =====================================
set(0, 'DefaultAxesFontSize', 18);
%%
figure(1);clf
subplot(2,1,1);
plot(1:T2,u(T1+(1:T2)));
xlabel('timesteps');
ylabel('input data');
title('input data')
xlim([1 T2])

subplot(2,1,2);
pcolor(1:T2,1:RC_N,M_train);shading interp;
xlabel('timesteps');
ylabel('node index');
title('non-optimised CCD output')
xlim([1 T2])

% gstitle('collect stage')

%% ========================================================================
figure(2);clf;

subplot(5,1,1)
pcolor(1:T2,1:RC_N,M_train);shading interp;
colorbar;
xlabel('timesteps');
ylabel('pixel index')
title('non-optimised CCD output')
xlim([1 T2])

subplot(5,1,2);
plot(Yhat_train);hold all;
plot(Y_train);hold off;
legend('data','best approximation');
xlabel('timesteps')
title('data');
xlim([1 T2]);

subplot(5,1,3);
semilogy(abs(Yhat_train-Y_train).^2);hold all;
title('error');
legend('data','best approximation');
xlabel('timesteps')

subplot(5,1,4);
plot(diff(Yhat_train));hold all;
plot(diff(Y_train));hold off;
title('derivative of data');
xlabel('timesteps');
legend('data','best approximation');
xlim([1 T2])

subplot(5,1,5);
semilogy(abs(diff(Yhat_train)-diff(Y_train)).^2);hold all;
legend('data','best approximation');
title('error');
xlabel('timesteps')

% gstitle(['training stage - NMSE=',num2str(fom_train,'%.2e')])
%% ========================================================================
figure(3);clf;

subplot(5,1,1)
pcolor(1:T3,1:RC_N,M_test);shading interp;
xlabel('timesteps');
ylabel('pixel index')
title('non-optimised CCD output')
colorbar;
xlim([1 T3])

subplot(5,1,2)
plot(Y_test);hold all;
plot(Yhat_test);hold off;
xlim([1 T3-1]);
xlabel('timesteps');
title('data');
legend('data','best approximation');
xlim([1 T3])

subplot(5,1,3)
semilogy(abs(Y_test-Yhat_test).^2);
xlim([1 T3-1]);
xlabel('timesteps');
title('error');
xlim([1 T3])

subplot(5,1,4)
plot(diff(Y_test));hold all;
plot(diff(Yhat_test));hold off;
xlim([1 T3-2]);
xlabel('timesteps');
title('derivative of data');
legend('data','best approximation');

subplot(5,1,5)
semilogy(abs(diff(Y_test)-diff(Yhat_test)).^2);
xlim([1 T3-2]);
xlabel('timesteps');
title('error');

% suptitle(['test stage  - NMSE=',num2str(fom_test,'%.2e')])
