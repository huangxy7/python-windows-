#import pyautogui
import numpy as np 
import cv2
import sys
import dlib
import numpy
from skimage import io
import sqlite3
import tkinter
import tkinter.messagebox
from PIL import Image
from tkinter import filedialog
#人脸识别器分类器
cas = cv2.CascadeClassifier("D://pythonface/haarcascade_frontalface_alt2.xml")   #载入级联分类器，即人脸数据库  
classfier=cv2.CascadeClassifier("D://pythonface/haarcascade_frontalface_default.xml")
facerec = dlib.face_recognition_model_v1("D://pythonface/dlib_face_recognition_resnet_model_v1.dat")
sp = dlib.shape_predictor('D://pythonface/shape_predictor_68_face_landmarks.dat')#加载检测器
detector = dlib.get_frontal_face_detector()
login=0
def facedetec():
    cv2.namedWindow("test")
    #1调用摄像头
    cap=cv2.VideoCapture(0)
    color=(0,255,0)
    while cap.isOpened():
        ok,frame=cap.read()
        if not ok:
            break
        #3灰度转换
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #人脸检测，图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:            #大于0则检测到人脸      
            print("检测到人脸")                             
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect  #5画图   
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3)#单独框出每一张人脸
        cv2.imshow("test",frame)#窗口显示
        if cv2.waitKey(10)&0xFF==ord('a'):#输入a截取图片并检测
            cv2.imwrite("D://pythonface/test.jpg", frame)#保存获取图片
            img = io.imread('D://pythonface/frist.jpg')#加载检测的图片
            detss = detector(img,1)# 图片人脸子描述
            if (len(detss) > 0):#大于0则检测到人脸
                for k,d in enumerate(detss):
                    shape = sp(img,d)
                    face_descriptor = facerec.compute_face_descriptor(img, shape)
                    v = numpy.array(face_descriptor) #本地图片子描述存进v
                img = io.imread("D://pythonface/test.jpg")
                dets = detector(img, 1)# 获取图片人脸子描述
                if (len(dets) > 0):
                    for k, d in enumerate(dets):
                        shape = sp(img, d)
                        face_descriptor = facerec.compute_face_descriptor(img, shape)
                        d_test = numpy.array(face_descriptor) #boy.jpg子描述存进d_test
                    # 计算欧式距离
                    dist = numpy.linalg.norm(v-d_test)
                    dist=(1-dist)*100    
                    print("相似度：",dist,"%")     
                    if dist>=70:
                        tkinter.messagebox.showinfo(title='提示信息',message='登录成功')
                        cap.release()
                        cv2.destroyAllWindows()#关闭窗口
                        root.destroy()#关闭登陆窗口
                        loginsuc()#调用登陆成功的函数
                    else:
                        tkinter.messagebox.showinfo(title='提示信息',message='登录失败')
                else:
                    tkinter.messagebox.showinfo(title='提示信息',message='没有检测到人脸')
        if cv2.waitKey(10)&0xFF==ord('e'):
            cv2.destroyAllWindows()#关闭窗口
            break

def picturrelogin():
    try:
        img = io.imread('D://pythonface/frist.jpg')#加载检测的图片
        # 获取图片人脸子描述
        faces = detector(img,1)# 图片人脸子描述
        for k,d in enumerate(faces):
            shape= sp(img,d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            des= numpy.array(face_descriptor) #本地图片子描述存进des
        #打开需要检测的图片
        mypicture=tkinter.filedialog.askopenfilename(title='Open Image',filetypes=[('image', '*.jpg *.png *.gif')])
        img = io.imread(mypicture)#加载检测的图片
        # 获取图片人脸子描述
        faces = detector(img,1)# 图片人脸子描述
        if (len(faces) > 0):#大于0则检测到人脸
                for k,d in enumerate(faces):
                        shape = sp(img,d)
                        face_descriptor = facerec.compute_face_descriptor(img, shape)
                        mypicdes= numpy.array(face_descriptor) #d登录图片子描述存进mypicdes
        # 计算欧式距离，算出人脸之间的差距
        dist = numpy.linalg.norm(des-mypicdes)
        dist=(1-dist)*100    
        print("相似度：",dist,"%")
        if dist>=70:
            tkinter.messagebox.showinfo(title='提示信息',message='登录成功')
            root.destroy()
            loginsuc()
        else:
            tkinter.messagebox.showinfo(title='提示信息',message='登录失败')     
    except:
        tkinter.messagebox.showinfo(title='提示信息',message='图片识别有错误没有或者大于一张人脸')
    
def log():
    roots=tkinter.Tk()#创建新窗口
    roots.title('密码登陆')
    roots['height']=250
    roots['width']=280
    varName=tkinter.StringVar('')
    varName.set('')
    varPwd=tkinter.StringVar('')
    varPwd.set('')
    labelName=tkinter.Label(roots,text='账号:',justify=tkinter.RIGHT,width=100)
    labelName.place(x=12,y=5,width=100,height=25)
    entryName=tkinter.Entry(roots,width=100,textvariable=varName)
    entryName.place(x=100,y=5,width=100,height=25)
    #创建账号和密码文本框
    labelPwd=tkinter.Label(roots,text='密码:',justify=tkinter.RIGHT,width=100)
    labelPwd.place(x=12,y=30,width=100,height=25)
    entryPwd=tkinter.Entry(roots,width=80,textvariable=varPwd)
    entryPwd.place(x=100,y=30,width=100,height=25)
    def mmlogins():
        try:
            name=entryName.get()
            pwd=entryPwd.get()
            conn=sqlite3.connect('data.db')#连接数据库
            c=conn.cursor()#创建cousor对象
            for rows in c.execute('SELECT*FROM namepa'):
                if rows[0]==name and rows[1]==pwd:
                    login=1#修改标志符状态为登陆成功
                    tkinter.messagebox.showinfo(title='提示信息',message='登录成功')
                    root.destroy()
                    roots.destroy()
                    loginsuc()
                    break
                else :
                    login=0#标志符改为登陆失败
            if login==0:
                tkinter.messagebox.showinfo(title='提示信息',message='账号或密码错误')
            conn.commit()#关闭数据库
            conn.close()#关闭数据库
        except:
            conn=sqlite3.connect('data.db')#连接数据库
            c=conn.cursor()#创建cousor对象
            c.execute('''create table namepa(name,pwd)''')
            name='huang'
            pwd='123456'
            conn=sqlite3.connect('data.db')#连接数据库
            c=conn.cursor()#创建cousor对象
            c.execute("INSERT INTO namepa values(?,?)",(name,pwd))#插入表初始数据
            for rows in c.execute('SELECT*FROM namepa'):
                print("账号：",rows[0],end=" ")
                print("密码：",rows[1])
            conn.commit()#关闭数据库
            conn.close()#关闭数据库
            tkinter.messagebox.showinfo(title='提示信息',message='这是第一次运行数据库，账号密码已创建。')
    buttoninpasswd=tkinter.Button(roots,text='登录',command=mmlogins)
    buttoninpasswd.place(x=30,y=80,width=80,height=30)
    def registration():
        rootss=tkinter.Tk()#创建新窗口
        rootss.title('注册')
        rootss['height']=250#窗口大小
        rootss['width']=280
        #创建标签
        labelName=tkinter.Label(rootss,text='新账号:',justify=tkinter.RIGHT,width=80)
        labelName.place(x=10,y=5,width=80,height=20)
        entryNames=tkinter.Entry(rootss,width=80,textvariable=varName)
        entryNames.place(x=100,y=5,width=80,height=20)
        #创建账号和密码文本框
        labelPwd=tkinter.Label(rootss,text='新密码:',justify=tkinter.RIGHT,width=80)
        labelPwd.place(x=10,y=30,width=80,height=20)
        entryPwds=tkinter.Entry(rootss,width=80,textvariable=varPwd)
        entryPwds.place(x=100,y=30,width=80,height=20)
        def newregistration():
            try:
                names=entryNames.get()#获取文本框信息
                pwds=entryPwds.get()
                conn=sqlite3.connect('data.db')#连接数据库
                c=conn.cursor()#创建cousor对象
                c.execute("INSERT INTO namepa values(?,?)",(names,pwds))#插入表数据
                for rows in c.execute('SELECT*FROM namepa'):
                    print("账号：",rows[0],end=" ")
                    print("密码：",rows[1])
                conn.commit()
                conn.close()
                tkinter.messagebox.showinfo(title='提示信息',message='注册成功')
                rootss.destroy()
            except:
                conn=sqlite3.connect('data.db')#连接数据库
                c=conn.cursor()#创建cousor对象
                c.execute('''create table namepass(name,pwd)''')#创建表
                namess=entryNamess.get()
                pwdss=entryPwdss.get()
                conn=sqlite3.connect('data.db')#连接数据库
                c=conn.cursor()#创建cousor对象
                c.execute("INSERT INTO namepa values(?,?)",(name,pwd))#插入表数据
                for rows in c.execute('SELECT*FROM namepa'):
                    print("账号：",rows[0],end=" ")
                    print("密码：",rows[1])
                conn.commit()
                conn.close()#关闭数据库
                tkinter.messagebox.showinfo(title='提示信息',message='注册成功')
                rootss.destroy()
        buttoninpasswdss=tkinter.Button(rootss,text='保存',command=newregistration)
        buttoninpasswdss.place(x=80,y=80,width=80,height=30)
    buttoninname=tkinter.Button(roots,text='注册',command=registration)
    buttoninname.place(x=120,y=80,width=80,height=30)


def openss():
    #选择要打开的图片
    filename = tkinter.filedialog.askopenfilename(title='选择要登录的图片',filetypes=[('image', '*.jpg *.png *.gif')])
    WIDTH=90
    HEIGHT=45
    ascii_char = list("-_+~<>i!lI;:,\"^`'.*oahkbdpqwm$@B%8&WM#ZO0QLCJUYXzcvunxrjft/\|()1{}[]? ")
    # 将256灰度映射到70个字符上
    def get_char(r,g,b,alpha = 256):
        if alpha == 0:
            return ' '
        length = len(ascii_char)
        gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)

        unit = (256.0 + 1)/length
        return ascii_char[int(gray/unit)]
    if __name__ == '__main__':
        im = Image.open(filename)
        im = im.resize((WIDTH,HEIGHT), Image.NEAREST)
        txt = ""
        for i in range(HEIGHT):
            for j in range(WIDTH):
                txt += get_char(*im.getpixel((j,i)))
            txt += '\n'
        print(txt)
        #字符画输出到文件 
        with open("output.txt",'w') as f:
            f.write(txt)
        tkinter.messagebox.showinfo(title='提示信息',message='转化成功')
        zhuanzifu.destroy()#关闭窗口
def facesb():#识别图片中的人脸函数
    filename = tkinter.filedialog.askopenfilename(title='选择要识别的图片',filetypes=[('image', '*.jpg *.png *.gif')])
    # 获取图片人脸子描述
    img = cv2.imread(filename)#载入一张包含人脸的图片  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    #检测人脸：跟数据库进行比较  
    #结果：人脸的坐标x, y, 长度, 宽度  
    rects = cas.detectMultiScale(gray)  
      
    for x, y, width, height in rects:  
        cv2.rectangle(img, (x, y), (x+width, y+height), (0,255, 0), 3)  
      
    cv2.imshow("face", img)  


#登陆成功后运行的函数
def loginsuc():
    suclogin=tkinter.Tk()#创建新窗口
    suclogin.title('图片转化系统')
    suclogin['height']=250
    suclogin['width']=280
    gongneng=tkinter.Label(suclogin,text='♥功能 :',justify=tkinter.RIGHT,width=80)#创建标签
    gongneng.place(x=30,y=30,width=80,height=40)
    def zifuhua():
        #创建加载图片的页面
        zhuanzifu=tkinter.Tk()#创建窗口
        zhuanzifu.title("转化为字符画")
        zhuanzifu['height']=240#窗口大小
        zhuanzifu['width']=280
        gongnengs=tkinter.Label(zhuanzifu,text='♥功能:将图片转为字符画',justify=tkinter.RIGHT,width=200)#创建标签
        gongnengs.place(x=30,y=30,width=200,height=40)
        def opens():
            try:#修改bug，当不想加载图片时关闭‘加载图片’的页面
                openss()
            except:
                zhuanzifu.destroy()
        zhuanbegin=tkinter.Button(zhuanzifu,text='加载图片',command=opens)
        zhuanbegin.place(x=80,y=80,width=90,height=40)
    zifuhuabut=tkinter.Button(suclogin,text='♥字符画功能',command=zifuhua)
    zifuhuabut.place(x=80,y=80,width=100,height=40)
    def facetest():
        renliansb=tkinter.Tk()
        renliansb.title("识别人脸")
        renliansb['height']=240
        renliansb['width']=280
        gongneng2=tkinter.Label(renliansb,text='♥功能:将图片里的人脸识别出来',justify=tkinter.RIGHT,width=200)#创建标签
        gongneng2.place(x=30,y=30,width=230,height=40)
        def openface():
            facesb()
        facesbbutton=tkinter.Button(renliansb,text='加载图片',command=openface)
        facesbbutton.place(x=80,y=80,width=90,height=40)
    facebutton=tkinter.Button(suclogin,text='识别人脸功能',command=facetest)
    facebutton.place(x=80,y=150,width=100,height=40)


root=tkinter.Tk()#创建窗口
root.title("图片管理系统")
root['height']=240#窗口大小
root['width']=280

#定义人脸登录函数
def sslogin():
    facedetec()
ssbutton=tkinter.Button(root,text='人脸登录',command=sslogin)
ssbutton.place(x=100,y=30,width=80,height=40)


#定义图片登录函数
def tplogin():
    picturrelogin()  
tpbutton=tkinter.Button(root,text='图片登录',command=tplogin)
tpbutton.place(x=100,y=100,width=80,height=40)


#定义密码登录函数
def mmlogin():
    log()
mmbutton=tkinter.Button(root,text='密码登录',command=mmlogin)
mmbutton.place(x=100,y=170,width=80,height=40)
root.mainloop()#窗口信息循环

