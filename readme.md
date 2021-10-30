##### Bilal Latif Ozdemir
## 5. Hafta Cevaplar
### 1-) One hot encoding modelin görmediği veriye nasıl uygulanır?
* Kategorik degiskenin bilinen seviyeleri bir yerde tutuluyor olur. Bu seviyelerden yogunluk olarak en fazla gozleme sahip olan ve/veya sonuc degiskeninde median degerinde yogunlasmis olan deger kaydedilir. Eger ki train setinde yer almayan bir sinif gozlemlenmis ise one-hot encoding yapmadan once yeni-yabanci sinifin median degerine donusturulmesi/replace islemi yapilir ve ardindan convert islemi yapilir.
```python
# oncesinde 'a', 'b', 'c', 'd' degerleri var oldugunu ve en fazla gozlenenin 'b' sinifi oldugunu varsayalim.
categorical_classes = ['a', 'b', 'c', 'd']
df['categorical_variable'] = df['categorical_variable'].apply(lambda x: x if x in categorical_classes else 'b')

encoder_one_hot.fit(df['categorical_variable'])
```
* Birr diger cozum de sklearn icerisindeki one-hot encoder metodunun _handle_unknown_ parametresini **_ignore_** olarak belirlemek olacaktir, ki bu bilinmeyen bir sinif geldiginde tum kolonlara 0 degerini atayacaktir.

### 2-) Labelencoding'de ilgili kolon için ölçeklendirme nasıl yapılır? (Verinin doğru etkisiyle sayısal olarak dönüştürülmesi)
- Label encoding sklearn icerisindeki hazir fonksiyon ile yapildigi zaman hatali durumlar olusabiliyor. Label encoding yaparken siniflarin orderlari belirtilmez ise (bu order mantiksal degil bagimli degisken ile sinifin oldugu desene gore ozellestirilebilir) rastgele atamalar makul olmayan bir sayisal sira elde edilmesine sebep olur. Bunun icin ilgili kategorik degiskenin siniflarinin bagimli degisken ile olusturdugu desen incelenmeli (istatistiksel testler ile yapilmasi onerilir fakat veri gorsellestirme yapilip karar verilmesi de yeterli olabilir) daha sonrasinda gerekli label encoder fonksiyonu el ile yazilmalidir. (hazir paketlerden bunu yapan var ise ben bilmiyorum).
- Ornek olmasi acisindan verisetimizdeki 'total_claim_amount' degiskeni bagimli degisken ve 'insured_relationship' ise label encode edecegimiz kategorik degiskenmiscesine gerekli islemleri yapiyor olacagim. Sadece mantigi gostericek olup diger detaylari atlayacagim.
```python
means = []
classes = []
for i in df['insured_relationship'].unique():
    mean = df[df['insured_relationship'] == i ]['total_claim_amount'].mean()
    means.append(mean)
    classes.append(i)
dfs = pd.DataFrame({
    'classes': classes,
    'means': means
})
dfs.sort_values(by='means', inplace=True)
dfs.reset_index(drop=True, inplace=True)

dfs
```
- Ornegin bu komutta elde ettigimiz cikti, ortalamalarin siralanmis hali bu sekilde:
![image](https://user-images.githubusercontent.com/70684994/139500364-9ad6d803-ce0d-4fd4-9eaf-06ddf8bd64e4.png)
Burada izleyecegim mantiga gore insured_relationship degiskeni siniflarindan **bagimli degiskende** en dusuk ortalamaya sahip olani 0, en yuksek ortalamaya sahip olani 5 olarak encode edecegim. (Detayli calismada ortalamalar arasi farkin anlamliligi da gozonunde alinmali)

```python
def encode_relationships(i):
    return int(dfs[dfs['classes'] == i].index.values)

for i in df['insured_relationship'].unique():
    print(f"{i} -----> {encode_relationships(i)}")
```
* cikti bu sekilde
![image](https://user-images.githubusercontent.com/70684994/139510299-6d5333cd-1ded-46ef-948e-859fd51aa3b2.png)

Encode islemini bu case'de yapacak fonksiyon da bu sekilde tanimlanabilir. Sonrasinda encode islemini kolona uyarlamak ise bu kadar basit:
```python
df['insured_relationship'] = df['insured_relationship'].apply(lambda x: encode_relationship(x))
```

### 3-) Imbalance datasette train test split yaparken neleri göz önünde bulundurmalıyız?
* Inbalanced datasetlerde en onemli kriter dengesizligin bir anlama yorumlanabilirligi veya yorumlanamamasi durumu. Mesela 10000 gozlemin oldugu bir verisetinde 5 farkli kategori sirasi ile 3000 2000 2000 2990 ve 10 gozlemde gorulmus ise burada 10 kadar gozlenen sinifin saptanabilirligi veya iliskisinin anlamliligini ortaya koymak makul degildir. Boyle bir durumda 10 gozlemde gorulmus sinifin onem duzeyine ve diger siniflar ile benzerligine gore bu sinifi dusurmek veya diger siniflar ile birlestirmek konusu degerlendirilebilir.
* Mesela 3000 gozleme sahip bir verisetinde 2 sinif sirasi ile 800 ve 2200 gozlemde gorulmus ise bu da bir dengesizlik durumudur. Boyle bir durumda mesela modelin ilk sinifla karsilasmasi ve bu sinifin yer aldigi grubu tahminlemesi icin ogrenmesi daha eksik kalacaktir. Bunun icin train test splitte bu dengesizligi train test setlerine esit bir sekilde dagitmak hatalarin onunu bir nebze alacaktir. Diger bir yaklasim da classifier'in eger ki olasiliksal bir tahminleme kullaniyor ise threshold noktasinda degisiklikler yapma tahminleme yuzdeleri uzerinde etkili olacaktir.

### 4-) Validation dataseti (modelin görmediği) nasıl oluşturulur ve nasıl predict etmeye hazır hale getirilir?
* Validation set modelin egitim suresi icinde gormedigi bir set olmasi gerekmekte ve ayni zamanda egitime katilan veriler ile benzer ozellikte oldugu varsayilan bir set olmaktadir. Bunun icin butun verisetini train test olarak ayirirken bu split islemini random gerceklestirmek bir yanliliga sebep olmamayi saglayacaktir ve gereklidir.
* Predict isleminde ise kritik nokta sudur, train setinde veri hangi islemlere tabii tutuldu ise test seti de birebir ayni islemlere tabii tutulmalidir. Daha sonrasinda predict islemi yapilmalidir.
### 5-) predict_proba metoduyla oran nasıl hesaplanır ve treshold nasıl değiştirilir?
- Bir classifier ile tahminleme isleminde 'predict' metodu kullanildiginda tahmin bir deger olarak "0, 2, 1" gibi doner. Fakat arkaplanda aslinda her mumkun durumla uyum icin bir olasilik ataniyor olup bize donen deger en yuksek olasiliga sahip degerdir.
  - Mesela 0, 1, 2, 3 durumlarini gormus ve tahmin etmesi beklenen model arkaplanda tahmin isleminde bu 4 durum icin sirasiyla 0.05, 0.15, 0.02, 0.78 gibi olasilik degerleri hesaplamis olur. Bu 4'luden en yuksek olasiliga sahip olan 0.78'in de karsilik geldigi degeri '3' u bizlere tahmin degeri olarak dondurur.
- Peki predict komutu degil de predict_proba metrodunu kullanirsak ne olur? Ornegin bir predict_proba ciktisi su sekilde olabilir:
```cmd
[array([[ 0.8,  0.2],
       [ 0.4,  0.6],
       [ 0.8,  0.2],
       [ 0.9,  0.1],
       [ 0.1,  0.9],
       [ 0.2,  0.8],
       [ 0.9,  0.1],
       [ 0.9,  0.1]])]
```
- Bu ciktida sagdaki 1, soldaki yuzdeler ise 0 durumunun olasiligidir. Default modelde threshold olarak 0.5 belirlenmistir ve bir olasilik durumu 0.5'in uzerinde ise baskin yuzde orada oldugu icin tahmin degeri olarak bu deger sunulur.
- Thresholdu degistirmek istedigimizde ise her 2 yuzdeyi ele almak isleri karistirabilir. Burada mesela toplamlarin 1 yaptigini bildigimiz icin soldaki yer alan degerler uzerinden gidilerek eger deger thresholdumuz uzerinde ise 0, degil ise 1 durumu tahmin degeri olarak dondurulebilir. Ornek kod ise:
```python
threshold = 0.4
pred_probs = model.predict_proba(X)
pred_probs.apply(lambda x: 0 if x[0] > threshold else 1)
```

### 6-) Fraud case'i üzerinde train&test&validation split, encoding, scaling,modelleme çalışmaları Python'da yapılarak, modelin görmediği dataset üzerinde başarılı sonuç alacak bir model örneği yapılmalı.
* Bu sorunun cevabi soru6.ipynb icerisinde yer aliyor. Notebookun ilk %60 kismi orjinal notebooktaki verionisleme kisimlarinin copy paste edilmis halleridir. Buradan sonraki kisim kendi yorumum ve cozumlerimden olusmaktadir.
