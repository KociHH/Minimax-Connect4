## Информация об игре

Игра — **"4 в ряд"** (аналог "Connect Four").

- Запуск идёт с `main.py`
- Цель: собрать **4 фишки в ряд** (по горизонтали, вертикали или диагонали)
- Вы играете **жёлтыми фишками**, AI — **красными**
- AI работает по алгоритму [Минимакс](https://ru.wikipedia.org/wiki/%D0%9C%D0%B8%D0%BD%D0%B8%D0%BC%D0%B0%D0%BA%D1%81):
  > Алгоритм перебирает все возможные ходы до заданной глубины, оценивает их и выбирает лучший, считая, что противник играет идеально.

---

## Установка и запуск

```bash
git clone https://github.com/KociHH/Minimax-Connect4.git
cd Minimax-Connect4

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

## Требования

Python 3.10 или выше