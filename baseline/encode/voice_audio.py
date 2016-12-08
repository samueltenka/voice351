#http://www.dyslexia-reading-well.com/44-phonemes-in-english.html
import scipy.io.wavfile as wav
import numpy

with open('s01-2/s0101a/s0101a.phones') as f:
    text = f.read()

lines = text.split('\n')
body = lines[9:]
times = [float(l.split()[0]) for l in body if l.split()]
f.close()

with open('s01-2/s0101a/s0101a.phones') as f:
    text = f.read()
lines = text.split('\n')
body = lines[9:]
phonemes = [str(l.split()[2]) for l in body if l.split()]
f.close()


(rate, sig) = wav.read('s01-2/s0101a/s0101a.wav')
D = {}
for i in range(0, len(phonemes)):
    if (i != len(phonemes) - 1):
        pho = sig[times[i] * rate : times[i + 1] * rate]
    if (phonemes[i] in D):
#        print [phonemes[i]]
        if (pho.size <= D[phonemes[i]].size):
            for j in range(0,pho.size - 1):
                pho[j] = (pho[j] + D[phonemes[i]][j]) / 2;
        else:
            for j in range(0,D[phonemes[i]].size - 1):
                pho[j] = (pho[j] + D[phonemes[i]][j]) / 2;
    D[phonemes[i]] = pho

    



#Seth = {'b': 'seth_b.wav', 'd': 'seth_d.wav', 'f': 'seth_f.wav', 'g': 'seth_g.wav', 'h': 'seth_h.wav', 'j': 'seth_j.wav', 'k': 'seth_k.wav', 'l': 'seth_l.wav', 'm': 'seth_m.wav', 'n': 'seth_n.wav', 'p': 'seth_p.wav', 'r': 'seth_r.wav', 's': 'seth_s.wav', 't': 'seth_t.wav', 'v': 'seth_v.wav', 'w': 'seth_d.wav', 'w': 'seth_w.wav', 'y': 'seth_y.wav', 'z': 'seth_z.wav', 'a': 'seth_a.wav', '20': 'seth_20.wav', 'e': 'seth_e.wav', '22': 'seth_22.wav', 'i': 'seth_i.wav', '24': 'seth_24.wav', 'o': 'seth_d.wav', '26': 'seth_26.wav', 'oo': 'seth_oo.wav', 'u': 'seth_u.wav', '29': 'seth_29.wav', '30': 'seth_30.wav', 'oi': 'seth_oi.wav', 'ow': 'seth_ow.wav', '33': 'seth_33.wav', '34': 'seth_34.wav', '35': 'seth_35.wav', '36': 'seth_36.wav', '37': 'seth_37.wav', '38': 'seth_38.wav', '39': 'seth_39.wav', 'zh': 'seth_zh.wav', 'ch': 'seth_ch.wav', 'sh': 'seth_sh.wav', 'th': 'seth_th.wav', 'ng': 'seth_ng.wav'}



# example
wav.write('test.wav', rate, sig[60*rate:65*rate ])

phonemelist = ['eh', 'k', 'r', 'ey', 'sh', 'n', 'el', 's', 'p', 'ow', 'r', 't', 's', 'aa', 'n', 's', 'ah', 'n', 'd', 'ey', 'ae', 'f', 't', 'er', 'n', 'uw', 'n']
output = D['r']
for i in phonemelist:
    output = numpy.append(output,D[i])
    
wav.write('averaged.wav', rate, output)
print "output", output



