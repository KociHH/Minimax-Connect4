from app.game_config import main

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Остановленно')