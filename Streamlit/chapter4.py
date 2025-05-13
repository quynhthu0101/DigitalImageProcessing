import numpy as np
import cv2
L = 256

def FrequencyFiltering(imgin, H):
    # Không cần mở rộng ảnh có kích thước PXQ
    f  = imgin.astype(np.float64)
    #Buoc1: DFT
    F = np.fft.fft2(f)
    #Buoc2: Shift vao the center of the inmage
    F = np.fft.fftshift(F)
    #Buoc3: Nhan F voi H
    G = F*H

    #Buoc4: Shift return, ta được G
    G = np.fft.ifftshift(G)

    #Buoc5: IDFT
    g = np.fft.ifft2(G)
    gR = np.clip(g.real, 0, L-1)
    imgout = gR.astype(np.uint8)

    return imgout

def Spectrum(imgin):
    M, N = imgin.shape
    #Buoc1
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P, Q), np.float32)
    #Buoc2
    fp[:M,:N] = 1.0*imgin/(L-1)

    #Buoc3
    for x in range(0, M):
        for y in range(0, N):
            if (x+y) % 2 == 1:
                fp[x, y] = -fp[x,y]
    #Buoc4
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)
    #Buoc5
    FR = F[:,:,0]
    FI = F[:,:,1]
    S = np.sqrt(FR**2 + FI**2)
    S = np.clip(S, 0, L-1)

    imgout = S.astype(np.uint8)
    return imgout

def CreateMoireFilter(M, N):
    H = np.ones((M, N), np.complex64)
    H.imag = 0.0
    
    u1, v1 = 44, 55
    u2, v2 = 85, 55
    u3, v3 = 41, 111
    u4, v4 = 81, 111

    u5, v5 = M - 44, N - 55
    u6, v6 = M - 85, N - 55
    u7, v7 = M - 41, N - 111
    u8, v8 = M - 81, N - 111

    D0 = 10
    for u in range(0, M):
        for v in range(0, N):   
            #u1, v1
            Duv = np.sqrt((u-u1)**2 + (v - v1)**2)
            if Duv <= D0:
                H.real[u, v] = 0.0
            #u2, v2
            Duv = np.sqrt((u-u2)**2 + (v - v2)**2)
            if Duv <= D0:
                H.real[u, v] = 0.0

            #u3, v3
            Duv = np.sqrt((u-u3)**2 + (v - v3)**2)
            if Duv <= D0:
                H.real[u, v] = 0.0

            #u4, v4
            Duv = np.sqrt((u-u4)**2 + (v - v4)**2)
            if Duv <= D0:
                H.real[u, v] = 0.0

            #u5, v5
            Duv = np.sqrt((u-u5)**2 + (v - v5)**2)
            if Duv <= D0:
                H.real[u, v] = 0.0

            #u6, v6
            Duv = np.sqrt((u-u6)**2 + (v - v6)**2)
            if Duv <= D0:
                H.real[u, v] = 0.0

            #u7, v7
            Duv = np.sqrt((u-u7)**2 + (v - v7)**2)
            if Duv <= D0:
                H.real[u, v] = 0.0

            #u8, v8
            Duv = np.sqrt((u-u8)**2 + (v - v8)**2)
            if Duv <= D0:
                H.real[u, v] = 0.0
    return H

def CreateInferenceFilter(M, N):
    H = np.ones((M, N), np.complex64)
    H.imag = 0.0
    D0 = 7
    D1 = 7
    for u in range(0, M):
        for v in range(0, N):
            if u not in range (M//2-D0, N//2+D0+1):
                if abs(v-N//2) <= D1:
                    H.real[u, v] = 0.0
    return H

def CreateMotionFilter(M, N):
    H = np.zeros((M, N), np.complex64)
    T = 1.0
    a = 0.1
    b = 0.1
    phi_prev = 0.0
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if abs(phi) < 1.0e-6:
                phi = phi_prev

            RE = T*np.sin(phi)*np.cos(phi)/phi
            IM = -T*np.sin(phi)*np.sin(phi)/phi

            H.real[u, v] = RE
            H.imag[u, v] = IM
            phi_prev = phi
    
    return H

def CreateDeMotionFilter(M, N):
    H = np.zeros((M, N), np.complex64)
    T = 1.0
    a = 0.1
    b = 0.1
    phi_prev = 0.0
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            mau_so = np.sin(phi)

            if abs(mau_so) < 1.0e-6:
                phi = phi_prev

            RE = phi / (T * np.sin(phi)) * np.cos(phi)
            IM = phi / T

            H.real[u, v] = RE
            H.imag[u, v] = IM
            phi_prev = phi
    
    return H

def RemoveMorie(imgin):
    M, N = imgin.shape
    H = CreateMoireFilter(M, N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout

def RemoveInterference(imgin):
    M, N = imgin.shape
    H = CreateInferenceFilter(M, N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout

def CreateMotion(imgin):
    M, N = imgin.shape
    H = CreateMotionFilter(M, N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout

def Demotion(imgin):
    M, N = imgin.shape
    H = CreateDeMotionFilter(M, N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout