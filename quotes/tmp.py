import os

def count_lines_in_files(folder_path):
    """
    Проходит по всем файлам .txt в указанной папке и выводит
    количество строк в каждом файле в формате:
    имя_файла, строк: <количество>
    """
    # Получаем список всех файлов в папке
    for filename in os.listdir(folder_path):
        # Обрабатываем только текстовые файлы с расширением .txt
        if not filename.lower().endswith(".csv"):
            continue

        file_path = os.path.join(folder_path, filename)

        # Проверяем, что это именно файл (а не папка)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    # Считаем количество строк
                    line_count = sum(1 for _ in f)
                print(f"{filename}, строк: {line_count}")
            except Exception as e:
                print(f"Не удалось прочесть файл {filename}: {e}")

if __name__ == "__main__":
    # Замените "data" на путь к вашей папке, если необходимо
    data_folder = "data"
    count_lines_in_files(data_folder)
