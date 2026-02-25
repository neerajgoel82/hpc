# Quick Start - Phase 5: Data Science

## Setup (5 minutes)

### 1. Create Virtual Environment
```bash
cd python/samples/phase5-datascience
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python 01-numpy/01_arrays_basics.py
```

## Learning Sequence

### Week 1: NumPy
```bash
cd 01-numpy
python 01_arrays_basics.py
python 02_indexing_slicing.py
# ... continue through all files
python exercises.py
```

### Week 2: Pandas
```bash
cd ../02-pandas
python 01_series_dataframes.py
# ... continue
```

### Week 3-8: Continue through modules

## Quick Examples

### NumPy Quick Test
```python
import numpy as np

# Create array
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Mean: {arr.mean()}")
```

### Pandas Quick Test
```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
})
print(df)
```

### Visualization Quick Test
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Sine Wave')
plt.show()
```

## Jupyter Notebook

To use Jupyter notebooks:
```bash
jupyter notebook
```

Then create new Python 3 notebook and run examples.

## Troubleshooting

**Issue**: `ModuleNotFoundError`
**Solution**: Make sure virtual environment is activated and dependencies are installed

**Issue**: Plot doesn't show
**Solution**: Add `plt.show()` or use `%matplotlib inline` in Jupyter

**Issue**: Memory error with large datasets
**Solution**: Use chunking with Pandas or reduce dataset size

## Next Steps

1. Complete all exercises in each module
2. Work on projects in `07-projects/`
3. Find real datasets on Kaggle
4. Move to Phase 6 (PyTorch) when ready

## Resources

- NumPy Docs: https://numpy.org/doc/
- Pandas Docs: https://pandas.pydata.org/docs/
- Matplotlib Gallery: https://matplotlib.org/stable/gallery/
- Kaggle Learn: https://www.kaggle.com/learn

Happy Learning! 📊
