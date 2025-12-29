import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium


# FUNKTIOT
#====================================================================================

# Alipäästösuodatin
from scipy.signal import butter, filtfilt
def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

#Haversinen kaava
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

#DATA
#=====================================================================================================

df_acc = pd.read_csv('https://raw.githubusercontent.com/JansQ-75/Soveltavan_Fysiikan_Loppuprojekti/main/loppuprojektiData/kiihtyvyys.csv')
df_gps = pd.read_csv('https://raw.githubusercontent.com/JansQ-75/Soveltavan_Fysiikan_Loppuprojekti/main/loppuprojektiData/gepsi.csv')

# Kiihtyvyys
data = df_acc['Linear Acceleration y (m/s^2)']  # askeleet näkyvät y-komponentissa
T_tot = df_acc['Time (s)'].max()                # datan pituus
n = len(df_acc['Time (s)'])                     # datapisteiden lukumäärä
fs = n/T_tot                                    # näytteenottotaajuus
nyq = fs/2                                      # nyqvistin taajuus
order = 3
cutoff = 1/0.4 # cutoff taajuus
data_filt = butter_lowpass_filter(data, cutoff, nyq, order) # suodatettu data

#GPS
#Rajataan datasta pois ne rivit, joilla horisontaalinen epätarkkuus on suuri
df_gps = df_gps[df_gps['Horizontal Accuracy (m)'] <4]
df_gps = df_gps.reset_index(drop=True)

#LASKENTA
#==============================================================================================

# Kokonaismatkan laskeminen

def kokonaismatka(df_gps):
    distances = []
    total_distance = 0
    for i in range(1, len(df_gps)):
        lat1 = df_gps.loc[i-1, 'Latitude (°)']
        lon1 = df_gps.loc[i-1, 'Longitude (°)']
        lat2 = df_gps.loc[i, 'Latitude (°)']
        lon2 = df_gps.loc[i, 'Longitude (°)']
        d = haversine(lon1, lat1, lon2, lat2) * 1000 # muunnetaan samalla metreiksi
        total_distance += d
        distances.append(d)
        
    
    df_gps['Distance_calc'] = [0] + distances   # Lisätään etäisyydet dataframeen
    df_gps['total_distance'] = df_gps['Distance_calc'].cumsum() #Lasketaan kokonaismatka mittapisteiden välisestä matkasta
    return total_distance

total_distance = kokonaismatka(df_gps)  # Kokonaismatkan pituus

# Askeleiden määrä suodatetusta kiihtyvyysdatasta

askeleet = 0
for i in range(n-1):
    if data_filt[i]/data_filt[i+1] <0:
        askeleet = askeleet + 1/2                           # Askeleiden määrä Fourier-analyysin perusteella
askeleenpituus = np.round((total_distance / askeleet),2)    # Askeleen pituus saadun askelmäärän perusteella  

# Askeleiden määrä Fourier-analyysin perusteella

dt = 1 / fs                     # oikea aika-askel näytteenottotaajuudesta
t = np.arange(0, T_tot, dt)

f = data_filt                   # käytetään suodatettua dataa

N = len(f)
fourier = np.fft.fft(f, N)

psd = fourier * np.conj(fourier) / N
freq = np.fft.fftfreq(N, dt)
L = np.arange(1, int(N/2))
PSD = np.array([freq[L], psd[L].real])

f_max = freq[L][psd[L] == np.max(psd[L])][0]
max_freq = np.round(f_max,2)
askeleetFFT = np.round(f_max * T_tot)   # Askeleiden määrä Fourier-analyysin perusteella
askeleenpituusFFT = np.round((total_distance / askeleetFFT),2)  # Askeleen pituus saadun askelmäärän perusteella
keskinopeus = df_gps['Velocity (m/s)'].mean()   # keskinopeus, eli nopeuden keskiarvo

# VISUALISOINTI
# =======================================================================================================================

st.title('Fysiikan loppuprojekti')
st.markdown("### _Janina Niemelä - TVTKMO24_")

st.text("Loppuprojektissa suoritettiin GPS- ja kiihtyvyyshavaintojen analyysi matkapuhelimen Phyphox -sovelluksen avulla.")
st.text("Analyysia varten tuli kävellä useamman minuutin ajan. Ensin käveltiin rauhallista vauhtia, seuraavaksi reipasta vauhtia ja lopuksi nopeampaa vauhtia (hölkkä/juoksu).")

st.header("Kiihtyvyysdata")

st.text("Kiihtyyvyysdatasta nähdään hyvin kävelyn 3 eri nopeutta")

# Kiihtyvyysdatan kuvaaja
fig, ax = plt.subplots()
ax.plot(df_acc['Time (s)'], data, label='Alkuperäinen data')
ax.plot(df_acc['Time (s)'], data_filt, label='Suodatettu data')
ax.set_title("Kiihtyvyysdata")
ax.set_xlabel('Time (s)')
ax.set_ylabel('Linear Acceleration y (m/s^2)')
ax.legend()

# Kiihtyvyysdatan kuvaaja zoomattuna
fig2, ax2 = plt.subplots()
ax2.plot(df_acc['Time (s)'], data, label='Alkuperäinen data')
ax2.plot(df_acc['Time (s)'], data_filt, label='Suodatettu data')
ax2.axis([135, 155, -10, 10])
ax2.set_title("Kiihtyvyysdata rajatulla ajanjaksolla")
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Linear Acceleration y (m/s^2)')
ax2.legend()

# Näytetään kuvaajat vierekkäin
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig)
with col2:
    st.pyplot(fig2)

# Kartta
st.header("Kartta")

st.text("Kuljettu matka piirrettynä kartalle.")

#Määritellään karttapohja, eli kartan keskipiste
lat1 = df_gps['Latitude (°)'].mean()    #leveysarvon keskiarvo
long1 = df_gps['Longitude (°)'].mean()    #pituusarvon keskiarvo

#luodaan kartta
my_map = folium.Map(location = [lat1, long1], zoom_start=17)

#piirretään reitti kartalle
folium.PolyLine(df_gps[['Latitude (°)', 'Longitude (°)']], color = 'red', weight = 3).add_to(my_map)
st_map = st_folium(my_map, width=900, height=650)

st.header("Matkan pituus")

st.text(f"Kokonaismatka: {total_distance:.2f} metriä")
st.markdown("_Matka laskettiin käyttäen Haversinen kaavaa_")

#matkan pituus ajan funktiona
fig3, ax3 = plt.subplots()
ax3.plot(df_gps['Time (s)'], df_gps['total_distance'])
ax3.set_title("Kuljettu matka")
ax3.set_xlabel('Aika (s)')
ax3.set_ylabel('Kokonaismatka (m)')
ax3.grid()
st.pyplot(fig3)

st.header("Askeleiden määrä, keskinopeus ja askeleen pituus")

st.subheader("Suodatetusta kiihtyvyysdatasta laskettuna:")
st.text(f"Askeleiden määrä: {askeleet: .0f}")
st.text(f"Askeleen pituus tällä askelmäärällä: {askeleenpituus: .2f} m")

st.subheader("Kiihtyvyysdatan Fourier-analyysin perusteella:")

#Tehospektri koko datasta
fig, ax = plt.subplots()
ax.plot(PSD[0,:], PSD[1,:])
ax.set_title(f'Tehospektri')
ax.axis([0,3,0,35000])
ax.set_xlabel('Taajuus (Hz)')
ax.set_ylabel('Teho')
ax.grid()
st.pyplot(fig)

st.text(f"Tehokkain taajuus, eli kävelytaajuus on {max_freq} Hz")
st.text(f"Askeleiden määrä: {askeleetFFT: .0f}")
st.text(f"Keskinopeus on {keskinopeus: .2f} m/s")
st.text(f"Askeleen pituus tällä askelmäärällä: {askeleenpituusFFT} m")

st.subheader("Otetaan huomioon 3 eri nopeutta:")

st.text("Kävelynopeus vaikuttaa askeleiden määrän ja pituuden tulkintaan. Tarkastellaan tätä tarkemmin.")

#Nopeus ajan funktiona
fig4, ax4 = plt.subplots()
ax4.plot(df_gps['Time (s)'], df_gps['Velocity (m/s)'])
ax4.set_title('Nopeus ajan funktiona')
ax4.set_xlabel('Aika (s)')
ax4.set_ylabel('Nopeus (m/s)')
ax4.grid()
st.pyplot(fig4)

st.text("Kuvaajan perusteella huomataan, että matkalla on liikuttu kolmella eri nopeudella")
st.text("Rauhallinen kävely: välillä 20-145s")
st.text("Reipas kävely: välillä 145-212s")
st.text("Hölkkä/juoksu: välillä 212-250s")


# Määritetään aikavälit kolmelle nopeudelle
time_seg1 = (df_acc['Time (s)'] >= 20) & (df_acc['Time (s)'] < 145) # rauhallinen kävely
time_seg2 = (df_acc['Time (s)'] >= 145) & (df_acc['Time (s)'] < 212)    # reipas kävely
time_seg3 = (df_acc['Time (s)'] >= 212) & (df_acc['Time (s)'] <= 250)   # hölkkä/juoksu

segments = [time_seg1, time_seg2, time_seg3]
segment_names = ['Rauhallinen kävely', 'Reipas kävely', 'Hölkkä/juoksu']    #Nimetään jaksot
total_steps = 0 #Kokonaisaskeleiden määrä

# For-loopin avulla aikasegmenttien huipputaajuus, askelten määrä ja pituus sekä keskinopeus
for i, (mask, name) in enumerate(zip(segments, segment_names)):
    data_seg = data_filt[mask]  # otetaan suodatetusta datasta vain halutun aikavälin osio
    if len(data_seg) > 0:
        # Fourier-muunnos ja tehospektri
        N = len(data_seg)
        dt_seg = 1 / fs
        fourier = np.fft.fft(data_seg, N)
        psd = fourier * np.conj(fourier) / N
        freq = np.fft.fftfreq(N, dt_seg)
        L = np.arange(1, int(N/2))
        PSD = np.array([freq[L], psd[L].real])
        
        # Lasketaan huipputaajuus
        freq_range = (PSD[0,:] >= 0.5) & (PSD[0,:] <= 3)    # Ihmisen askeltaajuus on välillä 0.5-3 Hz
        if np.any(freq_range):
            peak_index = np.argmax(PSD[1, freq_range])  # huipputaajuuden pitää siis olla edellä määritellyllä välillä
            peak_freq = PSD[0, freq_range][peak_index]  # kyseisen nopeuden huipputaajuus (=askeltaajuus)
            duration = df_acc['Time (s)'][mask].max() - df_acc['Time (s)'][mask].min()  # Aikavälin kesto
            steps_seg = peak_freq * duration    # aikavälin askelmäärä = askeltaajuus * kesto
            total_steps += steps_seg
            
            # Keskinopeus aikavälille
            start_time = df_acc['Time (s)'][mask].min() # aloitusaika kiihtyvyysdatasta
            end_time = df_acc['Time (s)'][mask].max()   # lopetusaika kiihtyvyysdatasta
            mask_gps = (df_gps['Time (s)'] >= start_time) & (df_gps['Time (s)'] <= end_time)    # gps-data kyseisellä välillä
            avg_speed = df_gps[mask_gps]['Velocity (m/s)'].mean()   # keskinopeus, eli nopeuden keskiarvo
            st.markdown(f"### {name}:")
            st.text(f"Askeltaajuus {peak_freq:.2f} Hz")
            st.text(f"Askeleiden määrä: {steps_seg:.0f}")
            st.text(f"Keskinopeus: {avg_speed:.2f} m/s")
            
            # Määritetään askeleen pituus
            if steps_seg > 0:
                #Askeleen pituus on matkan pituus jaettuna askeleiden määrällä
                step_length = (avg_speed * duration) / steps_seg
                st.text(f"Askelen pituus: {step_length:.2f} m")
        
        #Tehospektri aikaväliltä
        fig, ax = plt.subplots()
        ax.plot(PSD[0,:], PSD[1,:])
        ax.set_title(f'Tehospektri - {name}')
        ax.axis([0,3,0,35000])
        ax.set_xlabel('Taajuus (Hz)')
        ax.set_ylabel('Teho')
        ax.grid()
        st.pyplot(fig)

st.markdown(f"**Askeleiden määrä kolmesta eri nopeudesta laskettuna: {total_steps:.0f}**")

st.subheader("Pohdinta")
st.text("Kiihtyvyysdatasta voidaan onnistuneesti erottaa kolme eri nopeutta kävelysuorituksen aikana. Datan suodattaminen onnistui mielestäni hyvin")
st.text("Kartan piirtäminen onnistui myös hyvin. Datasta ei juurikaan tarvinnut suorittaa epätarkkuutta pois, sillä lähdin kävelemään vasta kun GPS-signaali oli hyvin saatavilla")
st.text("Matkan pituuden laskin käyttämällä Haversinen kaavaa. Vertailin tulosta mm. GoogleMapsin arvioon matkan pituudesta. Nämä täsmäsivät melko hyvin.")
st.markdown("Askeleiden määrä vaihteli laskentamenetelmän mukaan. Suoraan **suodatetusta datasta laskettuna saatiin askelmääräksi 504. Fourier-analyysin pohjalta laskettuna askelmäärä oli 456. Kun laskenta pilkottiin kolmeen osaan, eri nopeuksien mukaisesti, saatiin askelmääräksi 461**. " \
"Uskoisin, että viimeisin laskutapa antaa tarkimman tuloksen. Suoraan datasta laskettuna askelmäärään tulee ns. ylimääräisiä askeleita, kun sensori on virheellisesti tulkinnut liikahduksia askeleiksi. " \
"Kun taas tehdään Fourier-analyysi datasta ja lasketaan askelmäärä sen perusteella, päästään tarkempaan lopputulokseen. Tämä on kuitenkin tietyssä mielessä keskiarvo, sillä eri nopeuksia ei huomioida, vaan askeltaajuus on määritetty koko datan huipputaajuuden mukaan. " \
"Kun data pilkotaan kolmeen osaan eri nopeuksien mukaisesti, saadaan vielä tarkempi askeleiden lukumäärä.")
st.text("Myös askeleiden pituus vastaa paremmin todellisuutta, kun ne lasketaan eri nopeuksien mukaisesti. Askeleen pituus suodatetusta kiihtyvyysdatasta laskettuna oli 0.8m. " \
"Koko datan Fourier-analyysin perusteella askeleiden pituus oli 0.88m. " \
"Realistisin kuva askeleiden pituudesta saadaan, kun Fourier-analyysi tehdään erikseen jokaiselle nopeudelle. Rauhallisessa vauhdissa askelpituus oli 0.75m, reippaassa 0.82 ja juostessa 1.03m")