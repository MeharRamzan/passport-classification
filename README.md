# Passport Classification

### Installation

```
pip install -r requirements.txt
```
* Download model weights from [here](https://drive.google.com/file/d/1K5sWbItnEQosgVCFp2NcL57k8NqhTt0C/view?usp=sharing) and place them in passport classifier folder.
### Inference
```bash
python main.py PATH/TO/IMAGE
``` 
* Example 
```bash
python main.py sample_images/
```

### Integration

``` python
from model.classify import PassportClassifier
from PIL import Image

model = PassportClassifier(model_path='passport_classifier.pth')
pred = model.classify(img)
print(pred)
```
