### Grouped summary

```python
sales.groupby("color")["weight_kg"].mean()
```

### Pivot table

```python
dogs.pivot_table(values="weight_kg", index="color")
```

### Different statistics

```python
dogs.pivot_table(values="weight_kg", index="color", aggfunc=["mean", "median"])
```

### Pivot on two variables

```python
dogs.pivot_table(values="weight_kg", index="color", columns="breed", fill_value=0)
```

### Summing with pivot tables

```python
dogs.pivot_table(values="weight_kg", index="color", columns="breed", fill_value=0, margins=True)
```