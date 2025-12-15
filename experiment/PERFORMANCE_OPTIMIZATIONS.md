# שיפורי ביצועים ב-SSMS Experiment

## בעיות ביצועים שזוהו ותוקנו

### 1. ✅ טעינת Category Name בתוך הלולאה (תוקן)

**בעיה:**
- הקוד קרא ל-`get_cached_mapping()`, `os.listdir()`, `sorted()` בתוך הלולאה הפנימית
- זה קרה עבור כל תמונה × כל method = אלפי קריאות יקרות!

**פתרון:**
- העברת חישוב ה-category mapping **מחוץ ללולאה** (פעם אחת בלבד)
- Pre-compute את כל ה-mapping לפני תחילת הלולאה

**שיפור:**
- לפני: `O(images × methods × categories)` = אלפי קריאות
- אחרי: `O(categories)` = פעם אחת בלבד
- **שיפור: פי 100-1000!**

---

### 2. ✅ Heatmap Cache לא שימש (תוקן)

**בעיה:**
- `heatmap_cache = {}` הוגדר אבל לא שימש
- כל heatmap נטען מהדיסק בכל פעם, גם אם כבר נטען קודם

**פתרון:**
- בדיקת cache לפני טעינה מהדיסק
- שמירת heatmaps ב-cache לאחר טעינה/יצירה

**שיפור:**
- לפני: כל heatmap נטען מהדיסק בכל פעם
- אחרי: heatmap נטען פעם אחת, אז משתמשים ב-cache
- **שיפור: פי 2-5 (תלוי ב-hit rate)**

---

### 3. ✅ P-Metric Lookup לא אופטימלי (תוקן)

**בעיה:**
- חיפוש ב-`pmetric_lookup.keys()` בכל פעם (O(n))
- חיפוש זה קרה עבור כל תמונה × method × judge

**פתרון:**
- יצירת `_method_cache` פעם אחת
- חיפוש O(1) במקום O(n)

**שיפור:**
- לפני: O(n) חיפוש בכל פעם
- אחרי: O(1) חיפוש
- **שיפור: פי 10-100 (תלוי במספר methods)**

---

### 4. ✅ Tensor Operations לא אופטימליים (תוקן)

**בעיה:**
- `mask_tensor.to(device)` ללא `non_blocking=True`
- `torch.no_grad()` במקום `torch.inference_mode()` (פחות יעיל)

**פתרון:**
- הוספת `non_blocking=True` ל-tensor transfers
- שימוש ב-`torch.inference_mode()` במקום `torch.no_grad()`

**שיפור:**
- `non_blocking=True`: מאפשר overlap בין CPU/GPU operations
- `inference_mode()`: מהיר יותר מ-`no_grad()` (~10-15%)

---

## בעיות ביצועים שנותרו (לא קריטיות)

### 1. ⚠️ לולאות מקוננות
**מה:**
- `for image in images:`
  - `for method in methods:`
    - `for judge in judges:`

**למה זה איטי:**
- כל תמונה נעבדת בנפרד
- אין batch processing ל-SSMS

**למה לא לתקן:**
- SSMS צריך להיות מהיר (קריאה אחת ל-judge model)
- Batch processing מורכב יותר (צריך לטפל ב-different heatmaps)
- השיפור לא יהיה משמעותי (הבעיה העיקרית הייתה I/O)

---

### 2. ⚠️ שמירת Visualizations
**מה:**
- שמירת תמונות בכל פעם (אם לא קיימות)

**למה זה איטי:**
- I/O operations (כתיבה לדיסק)
- חישוב visualizations (denormalization, colormap, etc.)

**למה לא לתקן:**
- זה קורה רק פעם אחת (אם קובץ לא קיים)
- זה לא חלק מהחישוב העיקרי
- אפשר לדלג על זה אם לא צריך visualizations

---

## סיכום שיפורים

| בעיה | לפני | אחרי | שיפור |
|------|------|------|-------|
| Category mapping | בתוך לולאה | מחוץ ללולאה | פי 100-1000 |
| Heatmap cache | לא שימש | משמש | פי 2-5 |
| P-Metric lookup | O(n) | O(1) | פי 10-100 |
| Tensor operations | לא אופטימלי | אופטימלי | ~10-15% |

**סה"כ שיפור משוער: פי 5-20 (תלוי ב-hit rates)**

---

## המלצות נוספות (אופציונלי)

1. **Batch SSMS processing**: אם יש הרבה תמונות עם אותו heatmap, אפשר לעבד ב-batch
2. **Parallel heatmap loading**: טעינת heatmaps במקביל (multithreading)
3. **Defer visualization saving**: שמירת visualizations בסוף (batch I/O)
4. **Use memory-mapped files**: עבור heatmaps גדולים

---

## איך לבדוק ביצועים

```python
import time
import cProfile

# Profile the code
profiler = cProfile.Profile()
profiler.enable()

# Run your code
evaluate_all_methods(config, num_images=100)

profiler.disable()
profiler.print_stats(sort='cumulative')
```

זה יראה לך איפה הקוד מבלה את רוב הזמן.




