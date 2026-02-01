import os




def is_title(block: str) -> bool:
    return (len(block.split()) <= 10
            and block[0].isdigit() is False
            and block.endswith((".", ':')) is False)


#def create_chunks(text_blocks, check_titles):


def create_chuncks_from_folder(folder_path):
    full_db_text_blocks = []

    with os.scandir(folder_path) as files:
        for text_file in files:
            if text_file.is_file():
                with open(text_file.path, 'r', encoding='utf-8') as f:
                    text = f.read()

                blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

            full_db_text_blocks += blocks

    return full_db_text_blocks


if __name__ == '__main__':
    from rag.paths import SAMPLE_DATA_PATH

    full_db_text_blocks = create_chuncks_from_folder(SAMPLE_DATA_PATH)

    #check_titles = [is_title(block) for block in blocks[:50]]

    #print("")