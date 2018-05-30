import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.preprocessing import image



(train_x,train_y), (test_x,test_y) =mnist.load_data()

#train_x = train_x.astype('float32') / 255
#test_x = test_x.astype('float32') / 255


'''print(train_x)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)'''


train_x = train_x.reshape(60000,784)
test_x = test_x.reshape(10000,784)


train_y = keras.utils.to_categorical(train_y,10)
test_y = keras.utils.to_categorical(test_y,10)


#Initialize our data ************

model = Sequential()
model.add(Dense(units=128,activation="relu",input_shape=(784,)))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=10,activation="softmax"))


#*** Compile the model **********


model.compile(optimizer=SGD(0.001),loss="categorical_crossentropy",metrics=["accuracy"])


'''model.fit(train_x,train_y,batch_size=32,epochs=10,verbose=1)
accuracy = model.evaluate(x=test_x,y=test_y,batch_size=32)
print("Accuracy: ",accuracy[1])'''


#model.save("mnist-model.h5")

# load feautures after training our model
model.load_weights("mnist-model.h5")



img = image.load_img(path="4.jpeg",grayscale=True,target_size=(28,28,1))
img = image.img_to_array(img)
test_img = img.reshape((1,784))

img_class = model.predict_classes(test_img)
prediction = img_class[0]

classname = img_class[0]

print("Class: ",classname)
img = img.reshape((28,28))
plt.imshow(img)
plt.title(classname)
plt.show()
