%% FFT & CWT for time- and frequency-resolved spectroscopy
%% Pulse shape [ Gaussian*cos ]
sigma=0.07; % Width of the window
t=0:0.001:2;
Freq=1:0.5:100;
tau=1; % Center of the pulse wave
f=20; % Wave number in the window, approximately equivelent to the center frequency
PulseShape=1/(2*pi)^0.5/sigma.*exp(-(t-tau).^2./2./sigma.^2).*exp(1i*2*pi*f.*(t-tau));
subplot 331
hold off;
plot(t,cos(2*pi*f.*t),'-o','MarkerSize',2,'LineWidth',1.5,'Color',[0 0 0.8]);
hold on;
plot(t,real(PulseShape),'-o','MarkerSize',2,'LineWidth',1.5,'Color',[0.8 0 0]);
xlabel('t'); ylabel('Power'); title('Wave Shape of the Ultrafast Pulse');
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);
subplot 332
PulseShape_fft=fft(PulseShape);
plot(0:500*1/round(length(PulseShape_fft)/2-1):500,abs(PulseShape_fft(1:round(length(PulseShape_fft)/2))),'-o','MarkerSize',2,'LineWidth',1.5,'Color',[0.8 0 0]);
xlabel('\it f'); ylabel('Power'); title('Center Frequency');
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);

%% Signal generation
Signal=zeros(1,length(t));
decay_rate=[1, 2,  2,  0.1,  2,  0.2, 1.5, 0.2, 2];
frequency= [7, 20, 29,  30, 42,  44,  50,  66,  72]; % delta=5
for i=1:length(frequency)
    Signal=Signal+exp(-decay_rate(i).*t).*sin(2*pi*frequency(i).*t);
end
subplot 333
Time_Freq_spectr_real=zeros(length(Freq),length(t));
for idx_f=1:length(Freq)
    if length(find(frequency==Freq(idx_f)))==1
        Time_Freq_spectr_real(idx_f,:)=exp(-decay_rate(find(frequency==Freq(idx_f))).*t);
    end
end
imagesc(t,Freq,Time_Freq_spectr_real);
title('Real Time and Freq. Spectral');
colorbar('LineWidth',1.5);colormap('Jet');set(gca,'YDir','normal');
xlabel('t');ylabel('\it f');
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);
subplot 334
plot(t,Signal,'-o','MarkerSize',2,'LineWidth',1.5,'Color',[0.8 0 0]);
xlabel('t'); ylabel('Power'); title('Signal');
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);

%% Ordinary pulse (FFT)
Signal_fft=abs(fft(Signal));
subplot 335
plot(0:500*1/round(length(PulseShape_fft)/2-1):500,Signal_fft(1:round(length(PulseShape_fft)/2)),'-o','MarkerSize',2,'LineWidth',1.5,'Color',[0.8 0 0]);
xlabel('\omega'); ylabel('Power'); title('Signal after FFT');
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);

%% Ultrafast pulse (wave shape)
subplot 336
Time_Freq_spectr=zeros(length(Freq),length(t));
for idx_f=1:length(Freq)
for idx_t=1:length(t)
    tau=t(idx_t);
    f=Freq(idx_f);
    PulseShape=1/(2*pi)^0.5/sigma.*exp(-(t-tau).^2./2./sigma.^2).*exp(1i*2*pi*f.*(t-tau));
    Time_Freq_spectr(idx_f,idx_t)=PulseShape*Signal';
end
end
imagesc(t,Freq,imag(Time_Freq_spectr));
title('Heterodyne');
colorbar('LineWidth',1.5);colormap('Jet');set(gca,'YDir','normal');
xlabel('t');ylabel('\it f');
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);
subplot 337
imagesc(t,Freq,(abs(Time_Freq_spectr)).^2);
title('Homodyne');
colorbar('LineWidth',1.5);colormap('Jet');set(gca,'YDir','normal');
xlabel('t');ylabel('\it f');
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);

%% 1D slice with f
subplot 338
time=0;
t_slice=find(t==time);
hold off
plot(Freq,-max(max((Time_Freq_spectr(:,t_slice)).^2))*Time_Freq_spectr_real(:,t_slice),'-o','MarkerSize',2,'LineWidth',1.5,'Color',[0.8 0 0]);
hold on
plot(Freq,(abs(Time_Freq_spectr(:,t_slice))).^2,'-o','MarkerSize',2,'LineWidth',1.5,'Color',[0 0 0.8]);
xlabel('\it f');ylabel('Power'); title(['1D with t=',num2str(time)]);
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);

%% Combination of Ordinary and Ultrafast pulse
subplot 339
Homodyne=(abs(Time_Freq_spectr)).^2;
for idx_f=1:length(Freq)
    if length(find(frequency==Freq(idx_f)))==0
        Homodyne(idx_f,:)=0;
    end
end
imagesc(t,Freq,Homodyne);
title('Combination');
colorbar('LineWidth',1.5);colormap('Jet');set(gca,'YDir','normal');
xlabel('t');ylabel('\it f');
set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);

%% 1D slice with t
figure(2)
for i=1:length(frequency)
    subplot(3,3,i)
    hold off
    hold on
    plot(t,(Time_Freq_spectr(find(Freq==frequency(i)),:)),'-o','MarkerSize',2,'LineWidth',1.5,'Color',[0 0 0.8]);
    plot(t,500*Time_Freq_spectr_real(find(Freq==frequency(i)),:),'-o','MarkerSize',2,'LineWidth',1.5,'Color',[0.8 0 0]);
    xlabel('t');ylabel('Power'); title(['1D with f=',num2str(frequency(i)),' delay=',num2str(decay_rate(i))]);
    set(gca,'FontName','Times New Roman','FontSize',25,'LineWidth',1.5);
end
