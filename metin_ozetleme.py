import pandas as pd
import numpy as np
import re
import csv
import matplotlib.pyplot as plt
import spacy
from spacy.lang.tr import Turkish
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

nlp = Turkish()
nlp = spacy.blank("tr")

raw = []
with open('genel_veri_seti_metin.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    for row in reader:
        raw.append(row[0])

summary = []
with open('genel_veri_seti_baslik.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    for row in reader:
        summary.append(row[0])

pre = pd.DataFrame({'text': raw, 'summary': summary})

def text_strip(column):
    for row in column:
        row = re.sub("(İI)", "i", str(row)).lower()
        row = re.sub("(Şş)", "s", str(row)).lower()
        row = re.sub("(Ğğ)", "g", str(row)).lower()
        row = re.sub("(Üü)", "u", str(row)).lower()
        row = re.sub("(\s+)", " ", str(row)).lower()
        row = re.sub("(\s+.\s+)", " ", str(row)).lower()

        try:
            url = re.search(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", str(row))
            repl_url = url.group(3)
            row = re.sub(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", repl_url, str(row))
        except:
            pass

        yield row

processed_text = list(text_strip(pre['text']))
processed_summary = list(text_strip(pre['summary']))

text = [str(doc) for doc in nlp.pipe(processed_text, batch_size=32)]
summary = [' ' + str(doc) + ' ' for doc in nlp.pipe(processed_summary, batch_size=32)]

pre['cleaned_text'] = pd.Series(text)
pre['cleaned_summary'] = pd.Series(summary)

text_count = []
summary_count = []

for sent in pre['cleaned_text']:
    text_count.append(len(sent.split()))

for sent in pre['cleaned_summary']:
    summary_count.append(len(sent.split()))

graph_df = pd.DataFrame()
graph_df['text'] = text_count
graph_df['summary'] = summary_count
graph_df.hist(bins=5)
plt.show()

max_text_len = 150
max_summary_len = 50

cleaned_text = np.array(pre['cleaned_text'])
cleaned_summary = np.array(pre['cleaned_summary'])

short_text = []
short_summary = []

for i in range(len(cleaned_text)):
    if len(cleaned_summary[i].split()) <= max_summary_len and len(cleaned_text[i].split()) <= max_text_len:
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])

post_pre = pd.DataFrame({'text': short_text,'summary': short_summary})

post_pre.head(2)

# Özet için başlangıç ve bitiş kelimeleri belirleme

post_pre['summary'] = post_pre['summary'].apply(lambda x: 'sostok' + x \
        + ' eostok')

post_pre.head()

#Veri setlerini eğitim ve doğrulama için ayırdık

x_tr, x_val, y_tr, y_val = train_test_split(
    np.array(post_pre["text"]),
    np.array(post_pre["summary"]),
    test_size=0.1,
    random_state=0,
    shuffle=True,
)

#metinleri tokenizer ile işledik. x_tr kümesindeki veriler üzerinden eğitim yaptırdık

x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_tr))

#metinde ikiden az geçen kelimelerin yüzde kaç geçtiğinin bulunması
#nadir kelime yüzdesi yüksek ise odak arttırılması için nadir kelimeler filtrelenebilir.
thresh = 2

cnt = 0
tot_cnt = 0

for key, value in x_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    if value < thresh:
        cnt = cnt + 1
    
print(" Nadir kelimelerin yüzdesi: ", (cnt / tot_cnt) * 100)


#METNİ TOPLAM KELİME SAYISINA GÖRE YENİDEN TOKENİZE ETME METNİ SAYILARA DÖNÜŞTÜRÜP AYNI UZUNLUĞA GÖRE DOLDURMA
#Metin verilerini sayılara dönüştürdük ve verileri eşitlemek için daha kısa metinlerin kalan boşluklarını doldurma gibi gerekli işlemleri gerçekleştirdik.

x_tokenizer = Tokenizer(num_words = tot_cnt - cnt) 
x_tokenizer.fit_on_texts(list(x_tr))

x_tr_seq = x_tokenizer.texts_to_sequences(x_tr) 
x_val_seq = x_tokenizer.texts_to_sequences(x_val)

x_tr = pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')
x_val = pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

x_voc = x_tokenizer.num_words + 1

print("X cinsinden kelime dağarcığı boyutu = {}".format(x_voc))


# Özet için toplam kelime sayısına göre yeniden tokenize ettik. Özeti sayılara dönüştürüp aynı uzunluğa göre doldurduk.
# Metin için yaptığımız işlemlerin aynısını özet için de yaptık.Özet için de üçten az geçen kelimelerin sayısını bulduk.

y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))

thresh = 3

cnt = 0
tot_cnt = 0

for key, value in y_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    if value < thresh:
        cnt = cnt + 1
    
print("Nadir kelimelerin yüzdesi: ",(cnt / tot_cnt) * 100)


y_tokenizer = Tokenizer(num_words=tot_cnt-cnt) 
y_tokenizer.fit_on_texts(list(y_tr))

y_tr_seq = y_tokenizer.texts_to_sequences(y_tr) 
y_val_seq = y_tokenizer.texts_to_sequences(y_val) 

y_tr = pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
y_val = pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

y_voc = y_tokenizer.num_words + 1

print("Y cinsinden kelime dağarcığı boyutu = {}".format(y_voc))

#BOŞ ÖZETLERİ KALDIRMA

ind = []

for i in range(len(y_tr)):
    cnt = 0
    for j in y_tr[i]:
        if j != 0:
            cnt = cnt + 1
    if cnt == 2:
        ind.append(i)

y_tr = np.delete(y_tr, ind, axis=0)
x_tr = np.delete(x_tr, ind, axis=0)


#DOĞRULAMA VERİLERİ İÇİN DE BOŞ OLANLARIN KALDIRILMASI

ind = []
for i in range(len(y_val)):
    cnt = 0
    for j in y_val[i]:
        if j != 0:
            cnt = cnt + 1
    if cnt == 2:
        ind.append(i)

y_val = np.delete(y_val, ind, axis=0)
x_val = np.delete(x_val, ind, axis=0)

#AĞ MİMARİSİNİN LAYER,LSTM,ENCODER VE DECODER KISIMLARIYLA TANIMLANMASI.SEQ2SEQ MODELİNİN OLUŞTURULMASI
#Bu kod, metin özetleme için bir S2S modeli oluştururken kullandığımız temel yapıyı tanımlar. 
#Model daha sonra eğitilmesi ve metin özetleme görevi için uygun hale getirilecektir.

latent_dim =200
embedding_dim = 100


encoder_inputs = Input(shape=(max_text_len))


enc_emb = Embedding(x_voc, embedding_dim,
                    trainable=True)(encoder_inputs)


encoder_lstm1 = LSTM(latent_dim, return_sequences=True,
                     return_state=True, dropout=0.3,
                     recurrent_dropout=0.3)
(encoder_output1, state_h1, state_c1) = encoder_lstm1(enc_emb)


encoder_lstm2 = LSTM(latent_dim, return_sequences=True,
                     return_state=True, dropout=0.3,
                     recurrent_dropout=0.3)
(encoder_output2, state_h2, state_c2) = encoder_lstm2(encoder_output1)


encoder_lstm3 = LSTM(latent_dim, return_sequences=True,
                     return_state=True, dropout=0.3,
                     recurrent_dropout=0.3)
(encoder_output3, state_h3, state_c3) = encoder_lstm3(encoder_output2)


encoder_lstm4= LSTM(latent_dim, return_state=True,
                     return_sequences=True, dropout=0.3,
                     recurrent_dropout=0.3)
(encoder_outputs, state_h, state_c) = encoder_lstm4(encoder_output3)


decoder_inputs = Input(shape=(None,))


dec_emb_layer = Embedding(y_voc, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)


decoder_lstm = LSTM(latent_dim, return_sequences=True,
                    return_state=True, dropout=0.3,
                    recurrent_dropout=0.2)
(decoder_outputs, decoder_fwd_state, decoder_back_state) = \
    decoder_lstm(dec_emb, initial_state=[state_h, state_c])


decoder_dense = TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

#MODELİ EĞİTME
#MODELİN DERLENMESİ VE KULLANILACAK OLAN PARAMETRELERİN AYARLANMASI
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=5)


#MODELİN EĞİTİMİ İŞLEMLERİ 
#
history = model.fit(
    [x_tr, y_tr[:, :-1]],                                               #GİRİŞ VE ÇIKIŞ VERİLERİ
    y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:],               #HEDEF VERİLERİN YENİDEN ŞEKİLLENDİRİLMESİ   
    epochs=50,                                                           #EĞİTİM DÖNGÜSÜNÜN TEKRARLANMA SAYISI
    callbacks=[es],                                                     #GERİ ARAMA İŞLEVİ İÇİN EARLY STOPPİNG KULLANILIYOR.
    batch_size=32,                                                      #HERBİR EĞİTİMDE KULLANILACAK ÖRNEK SAYISI
    validation_data=([x_val, y_val[:, :-1]],                            #DOĞRULAMA VERİLERİ VE HEDEFLER 
                     y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:
                     , 1:]),)

# eğitim süreci boyunca modelin kaybını (loss) görselleştirmek için kullanılır. history.history['loss'] ve history.history['val_loss'] ifadeleri, 
# eğitim ve doğrulama veri setleri için kayıp değerlerini içeren geçmiş nesnesinin ilgili özelliklerini temsil eder.
# eğitim ve doğrulama süreçlerindeki kayıp değerlerin karşılaştırılması ve modelin performansını değerlendirmek için kullanıldı.
from matplotlib import pyplot

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


#reverse_target_word_index değişkeni, hedef dildeki kelime endekslerini kelimelere dönüştürmek için kullanılan bir sözlük yapısını temsil eder. 
# Bu sözlük, hedef dildeki kelime endekslerini anahtar olarak kullanarak gerçek kelimeleri elde etmek için kullanılır. Örneğin, reverse_target_word_index[3] ifadesi, 
# hedef dildeki 3 numaralı kelime endeksinin karşılık gelen gerçek kelimeyi döndürecektir.
#modelin eğitimi sırasında tahminleri gerçek kelimelere dönüştürmek ve değerlendirmek için kullanılır.

reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index


#Bu yapılar, modelin eğitimi sırasında ve tahmin yaparken encoder ve decoder katmanlarını ayrı ayrı kullanmak ve durumları koruyarak ilerlemek için kullanılır.

encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs,
                      state_h, state_c])




decoder_state_input_h = Input(shape=(latent_dim, ))
decoder_state_input_c = Input(shape=(latent_dim, ))
decoder_hidden_state_input = Input(shape=(max_text_len, latent_dim))


dec_emb2 = dec_emb_layer(decoder_inputs)

(decoder_outputs2, state_h2, state_c2) = decoder_lstm(dec_emb2,
        initial_state=[decoder_state_input_h, decoder_state_input_c])


decoder_outputs2 = decoder_dense(decoder_outputs2)


decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input,
                      decoder_state_input_h, decoder_state_input_c],
                      [decoder_outputs2] + [state_h2, state_c2])



#Bu fonksiyon, modelin tahmin yapması için kullanılır. Giriş dizisini kodlayarak bir özet oluşturur ve tahmin sürecini durdurma koşullarına göre yönetir.
def decode_sequence(input_seq):

   
    (e_out, e_h, e_c) = encoder_model.predict(input_seq)

   
    target_seq = np.zeros((1, 1))

    
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ' '

    while not stop_condition:
        (output_tokens, h, c) = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != 'eostok':
            decoded_sentence += ' ' + sampled_token

      
        if sampled_token == 'eostok' or len(decoded_sentence.split()) >= max_summary_len - 1:
            stop_condition = True

        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        
        (e_h, e_c) = (h, c)

        decoded_sentence = decoded_sentence.strip()
    return decoded_sentence

#seq2summary()ve seq2text()bunlar sırasıyla özet ve metnin sayısal temsilini dize temsiline dönüştürür.


def seq2summary(input_seq):
    newString = ''
    for i in input_seq:
        if i != 0 and i != target_word_index['sostok'] and i != target_word_index['eostok']:
            newString = newString + reverse_target_word_index[i] + ' '

    return newString


def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if i != 0:
            newString = newString + reverse_source_word_index[i] + ' '
  
    return newString

for i in range(0, 20):
    print ('Metin:', seq2text(x_tr[i]))
    print ('Orijinal özet:', seq2summary(y_tr[i]))
    print ('Tahmini özet:', decode_sequence(x_tr[i].reshape(1,max_text_len)))
    print ('\n')
    