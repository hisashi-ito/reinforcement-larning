'''
カメラテスト用プログラム
Copyright(c) 2020 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
'''
import cv2

def main():
  cap = cv2.VideoCapture(0)
  while True:
    ret, frame = cap.read()#画像の読み込み
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#グレースケールに変換
    cv2.imshow('gray', gray)#画像表示
    key = cv2.waitKey(10)#キー入力
    print(key)
    if key == 115:#sキーの場合
      cv2.imwrite('camera.png', gray)#画像保存
    elif key == 113:#qキーの場合
      break#ループを抜けて終了
  cap.release()

if __name__ == '__main__':
  main()
