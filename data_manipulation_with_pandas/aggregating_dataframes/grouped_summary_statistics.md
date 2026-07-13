# Summaries by group

### Invidual summary
```python
dogs[dogs["color"] == "Black"]["weight_kg"].mean()
dogs[dogs["color"] == "Brown"]["weight_kg"].mean()
dogs[dogs["color"] == "White"]["weight_kg"].mean()
dogs[dogs["color"] == "Gray"]["weight_kg"].mean()
dogs[dogs["color"] == "Tan"]["weight_kg"].mean()
```

### Grouped summaries
```python
dogs.groupby("color")["weight_kg"].mean()
```

### Multiple Grouped summaries
```python
dogs.groupby("color")["weight_kg"].agg([min, max, sum])
```

### Multiple Grouped Variables
```python
dogs.groupby(["color", "breed"])["weight_kg"].mean()
```
### Many groups, Many summaries
```python
dogs.groupby(["color", "breed"])[["weight_kg", "height_cm"]].mean()
```