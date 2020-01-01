<p align="center">
<img src="https://user-images.githubusercontent.com/8604827/71643004-fb6ca600-2cf6-11ea-8dd5-daba76687088.png" width="400px">
</p>

#### 1. はじめに  
本リポジトリは「強化学習と深層学習 C言語によるシミュレーション」小高 知宏著に記載されているコードの模写,Python化のコードを保存するrepositoryとする。

#### 2. ディレクトリ構成
* RL

#### 3. git clone 方法
* ~/.netrc にユーザ名/パスワードを書く
```bash
machine github.com
login username
password xxxxxxx
```
* clone  
submoduleを利用しているのでrecursiveのフラグを利用する。  
```bash
$ git clone https://github.com/hisashi-ito/reinforcement-larning.git
```

* ネットワーク設定が完了しているが、通信できない場合は以下のnssをupdatする。  
```bash
$ sudo yum update -y nss
```
