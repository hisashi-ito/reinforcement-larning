/*
  ネズミ学習問題用スケッチ（全部入り）
  Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
*/
#include <Servo.h>

Servo myservo;
int state;//自販機のON：1，OFF：0
unsigned long prev_t;//ボタンが押されてからの経過時間

void setup() {
  pinMode(4, INPUT); //電源スイッチ
  pinMode(5, INPUT); //商品スイッチ
  pinMode(6, OUTPUT); //電源LED
  myservo.attach(9);//サーボモータの設定
  myservo.write(60);//初期角度に移動
  prev_t = millis();
  state = 0;//電源をOFF
  digitalWrite(6, state); //最初は消灯
}

void loop() {
  if (digitalRead(4) == LOW) { //電源ボタンが押されたら
    if (state == 0)state = 1; //ON・OFF反転
    else state = 0;
    digitalWrite(6, state); //stateに従ってLEDの点灯・消灯
    delay(2000);//2秒待つ
    prev_t = millis();//現在の時間で経過時間をリセット
  }
  if (digitalRead(5) == LOW) { //商品ボタンが押されたら
    if (state == 1) { //電源がONなら
      myservo.write(120);//商品用のRCサーボモータを回転
      delay(1000);//1秒待つ
      myservo.write(60);//初期角度に
    }
    prev_t = millis();//現在の時間で経過時間をリセット
  }
  if (millis() - prev_t > 2500) { //2.5秒経過したか？
    prev_t = millis(); //現在の時間で経過時間をリセット
    state = 0;//電源をOFF
    digitalWrite(6, state); //stateに従ってLEDの点灯・消灯
  }

}
