%%              Simple OFDM TX and RX simulation

%%% NO ML %%%

clear;
close all;
clc;


%%              ADJUSTABLE PARAMETERS

M = 16; % Modulation order
N_FFT = 64; % IFFT/FFT size (number of subcarriers)
CPrefix = N_FFT/8; % Length of Cyclic Prefix
N_Pilot = 16; % Number of pilot symbols
snrVec = (1:1:30)'; % SNR values to loop through
N_taps = 4; % Number of channel taps

%%             TRANSMITTER

N_Data = N_FFT - N_Pilot; % Number of data carriers
k = log2(M); % Number of bits per symbol
p_sym = 0.948683298050514 + 0.948683298050514i; % pilot symbol
berEst = zeros([size(snrVec), 4]); % Vector used to store BER estimates
N_bits = k * N_Data; % Number of random bits to send
test_data_8pilots = [];

for n = 1:length(snrVec)
    numErrs = 0;
    numErrsNoeq = 0;
    numErrsZF = 0;
    numErrsMMSE = 0;
    numErrsIdeal = 0;
    numBits = 0;
    SNR = snrVec(n);
    while numBits < 1e4
        
        % Amount of data carriers
        
        PI = (N_FFT-2) / (N_Pilot-2);  % Pilot interval
        Ip = [1, floor(PI:PI:N_FFT), N_FFT];   % Index of pilots
        Id = setxor(1:N_FFT, Ip); % Index of data

        
        % Generate vector of random binary data
        tx_bits = randi([0 1], 1, N_bits);
        
        % Reshape and modulate
        tx_mod = reshape(tx_bits, length(tx_bits) / k, k);
        tx_mod= bi2de(tx_mod);
        tx_mod_training = zeros(64, 1); % Used for ML training
        tx_mod_training(Id) = tx_mod;
        tx_mod_training(Ip) = 8;
        tx_mod = qammod(tx_mod, M, 'gray', 'UnitAveragePower', true);
        avgPower = mean(abs(tx_mod).^2);
        
        % Fill data and pilot bins, generate OFDM signal
        tx_bins = zeros(N_FFT, length(tx_mod) / N_Data);
        tx_bins(Ip) = p_sym;
        tx_bins(Id) = tx_mod;
        tx_bins = reshape(tx_bins, [N_FFT, length(tx_bins) / N_FFT]);
        tpilots = tx_bins(Ip);
        tx_bins_time = ifft(tx_bins, N_FFT);
        
        % Add cyclic prefix
        tx_bins_cp = [tx_bins_time(end-CPrefix+1:end, :); tx_bins_time];
        
        
        %%          CHANNEL
        

        % Add multipath
        
        
        channel = randn(1,N_taps) + 1j * randn(1,N_taps);
        channel = channel./norm(channel); % normalization
        tx_bins_cp = filter(channel, 1, tx_bins_cp);
        
        
        channel_frequency_response = fft(channel, N_FFT).'; % True channel frequency response
        

        % Add noise
        noise = 1 / sqrt(2) * (randn(length(tx_bins_cp), 1) + 1j * randn(length(tx_bins_cp), 1));
        snr_lin = 10^(SNR / 10);
        noise = noise / sqrt(snr_lin) * rms(tx_bins_cp);
        tx_bins_cp_noise = tx_bins_cp + noise;
       
        
        %%          RECEIVER
        
        % Parallel to serial
        tx_bins_cp_noise = reshape(tx_bins_cp_noise, 1, N_FFT+CPrefix);
        
        % Serial to parallel
        tx_bins_cp_noise = reshape(tx_bins_cp_noise, N_FFT+CPrefix, length(tx_bins_cp_noise) / (N_FFT+CPrefix));
        
        % Remove cyclic prefix
        tx_bins_cp_noise(1:CPrefix, :) = [];
        rx_bins_noise = tx_bins_cp_noise;
        
        % Time to frequency domain
        rx_bins_noise = fft(rx_bins_noise, N_FFT);
        
        % Estimate channel
        rpilots = rx_bins_noise(Ip); % Received pilot symbols

        % Zero-forcing        
        hEst = rpilots./tpilots;
        hEstZfTemp = interp1(Ip, hEst, 1:N_FFT, 'spline')';
        hEstZF = conj(1 ./ hEstZfTemp(Id));
            
        % MMSE
        noiseVar = mean(abs(noise).^2);
        hEstMMSE = conj(conj(hEstZfTemp(Id))./(conj(hEstZfTemp(Id)).*hEstZfTemp(Id) + noiseVar));
        
        % Full CSI knowledge
        hEstIdeal = 1 ./ channel_frequency_response(Id);

       % Channel |H| estimates
%         plot(1:N_Data, abs(hEstIdeal), 'Color', 'blue')
%         hold on;
%         plot(1:N_Data, abs(hEstZF), 'Color', 'red')
%         hold on;
%         plot(1:N_Data, abs(hEstMMSE), 'Color', 'black')
%         hold on;
%         legend('Full CSI knowledge', 'ZF', 'MMSE')
%         xlabel('Subcarrier N')
%         ylabel('1 / |H(f)|')
        %title('Channel frequency response equalization coefficients estimation, SNR 30, 16 pilots')

        test_data_8pilots = [test_data_8pilots; real(rx_bins_noise), imag(rx_bins_noise), real(hEstZfTemp), imag(hEstZfTemp)*-1, tx_mod_training];

        % Remove pilot bins
        rx_bins_noise(Ip) = [];
        
        % Equalize channel
        rx_bins_eqZF = rx_bins_noise .* hEstZF;
        rx_bins_eqMMSE = rx_bins_noise .* hEstMMSE;
        rx_bins_ideal = rx_bins_noise .* hEstIdeal;

        
        % Demodulate full CSI knowledge
        rx_demodIdeal = qamdemod(rx_bins_ideal, M, 'gray', 'UnitAveragePower', true);
        rx_demodIdeal = de2bi(rx_demodIdeal, k);
        rx_bitsIdeal = rx_demodIdeal(:)';
        
        % Demodulate No-eq
        rx_demodNoeq = qamdemod(rx_bins_noise, M, 'gray', 'UnitAveragePower', true);
        rx_demodNoeq = de2bi(rx_demodNoeq, k);
        rx_bitsNoeq = rx_demodNoeq(:)';

        % Demodulate ZF
        rx_demodZF = qamdemod(rx_bins_eqZF, M, 'gray', 'UnitAveragePower', true);
        rx_demodZF = de2bi(rx_demodZF, k);
        rx_bitsZF = rx_demodZF(:)';
        
        % Demodulate MMSE
        rx_demodMMSE = qamdemod(rx_bins_eqMMSE, M, 'gray', 'UnitAveragePower', true);
        rx_demodMMSE = de2bi(rx_demodMMSE, k);
        rx_bitsMMSE = rx_demodMMSE(:)';


        % Calculate biterror

        % Zero-forcing equalization error
        [num_errZF, berZF] = biterr(tx_bits, rx_bitsZF);

        % MMSE equalization error
        [num_errMMSE, berMMSE] = biterr(tx_bits, rx_bitsMMSE);

        % No equalization
        [num_errNoeq, berNoeq] = biterr(tx_bits, rx_bitsNoeq);

        %Full CSI knowledge error
        [num_errIdeal, berIdeal] = biterr(tx_bits, rx_bitsIdeal);



        numErrs = numErrs + num_errNoeq;
        numErrsNoeq = numErrsNoeq + num_errNoeq;
        numErrsZF = numErrsZF + num_errZF;
        numErrsMMSE = numErrsMMSE + num_errMMSE;
        numErrsIdeal = numErrsIdeal + num_errIdeal;
        numBits = numBits + N_bits;
    
    end
    berEst(n, 1) = numErrsNoeq/numBits;
    berEst(n, 2) = numErrsZF/numBits;
    berEst(n, 3) = numErrsMMSE/numBits;
    berEst(n, 4) = numErrsIdeal/numBits;
end

semilogy(snrVec, berEst(:, 1), '--+')
hold on
semilogy(snrVec, berEst(:, 2), '--o')
hold on
semilogy(snrVec, berEst(:, 3), '--s')
hold on
semilogy(snrVec, berEst(:, 4), 'Color', 'green')
hold on
grid
legend('No-EQ', 'ZF', 'MMSE', 'Full CSI knowledge')
xlabel('SNR (db)')
ylabel('Bit Error Rate')
