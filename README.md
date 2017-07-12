# ResNet-152 for Keras
Adaptation of ResNet-152 to match Keras API with added large input option. Original code from [flyyufelix](https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6), Keras 2.0 modified version of original code from [mvoelk](https://gist.github.com/mvoelk/ef4fc7fb905be7191cc2beb1421da37c).

Compatible with both TensorFlow and Theano backends.

## Examples

This version is modified to work the same as the ResNet50 model currently available in `keras.applications`

Examples as provided by [fchollet](https://github.com/fchollet/deep-learning-models) are compatible as shown below:

### Classify images

```python
from resnet152 import ResNet152
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

model = ResNet152(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print(‘Predicted:’, decode_predictions(preds))
#Predicted: [[('n02504458', 'African_elephant', 0.57481891)]
```

### Extract pool5 features from images

```python
from resnet152 import ResNet152
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

model = ResNet152(include_top=False, weights='imagenet')
    
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
    
features = model.predict(x)
```

### Extract 14x14x2048 features from res5c

```python
from resnet152 import ResNet152
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model

base_model = ResNet152(weights='imagenet', large_input=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('res5c').output)
    
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(448,448))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
    
res5c_features = model.predict(x)
```

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [ResNet50 in Keras](https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py)
- [Original ResNet152 Keras code by flyyufelix](https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6)
  - [Keras 2.0 version modifications by mvoelk](https://gist.github.com/mvoelk/ef4fc7fb905be7191cc2beb1421da37c)
