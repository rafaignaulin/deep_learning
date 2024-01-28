import os
import pathlib
import shutil


main_path = pathlib.Path(r"./GroceryStoreDataset/dataset")

path_train = os.path.join(main_path,"train")
path_test = os.path.join(main_path,"test")
path_val = os.path.join(main_path,"val")


# %%
# Remoção de subniveis dos diretórios (parse de classes granulares, como tipo de frutas)
def preprocess():
    level = 0
    def verifyDir(dir, last_dir= None, level=0):
        subdir = os.listdir(dir)
        level += 1
        for file in subdir:
            subpath = os.path.join(dir, file)
            if os.path.isdir(subpath):
                verifyDir(subpath, dir, level)
            else:

                if level == 4:
                    shutil.move(subpath, os.path.join(last_dir, file))
                    if len(os.listdir(dir)) == 0:
                        shutil.rmtree(dir)

    for path in [path_train, path_test, path_val]:
        verifyDir(path, None)

    # %%
    # Organização dos subdiretorios das classes e eliminação das classes primarias (frutas, pacotes e vegetais)

    col = ["Fruit", "Packages", "Vegetables"]
    for path in [path_train, path_test, path_val]:
        for c in col:
            caminho = os.path.join(path, c)
            print(caminho)
            for dir in os.listdir(caminho):
                print(os.path.join(caminho, dir),  len(os.listdir(os.path.join(caminho))))
                shutil.move(os.path.join(caminho, dir), path)
                # if len(os.listdir(caminho)) == 1:
                #     shutil.rmtree(caminho)

    for path in [path_train, path_test, path_val]:
        for c in col:
            shutil.rmtree(os.path.join(path, c))

    # %%
    # Olhar intersecção de classes, para identificar classes que estão no treino, porém não estão no conjunto de teste ou validação (e vice versa)

    dc = {}
    for path in [path_train, path_test, path_val]:
        a = []
        for file in os.listdir(path):
            a.append(file)
            dc[path.split("/")[-1]] = set(a)

        intersect = dc["train"].intersection(dc["test"], dc["val"])

    # %%
    # Remoção das classes restantes (que estão somente no treino, ou somente no teste)

    for path in [path_train, path_test, path_val]:
        for file in os.listdir(path):
            if file not in intersect:
                print(os.path.join(path, file))
                shutil.rmtree(os.path.join(path, file))
