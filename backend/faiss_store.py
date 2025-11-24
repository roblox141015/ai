# 簡易 FAISS 向量儲存與查詢 helper
import faiss
import numpy as np
import os
import pickle

class FaissStore:
    def __init__(self, dim, index_path="faiss.index", meta_path="data_store.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.meta = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.meta = {"ids": [], "docs": []}  # parallel arrays

    def add(self, vecs, docs, ids=None):
        arr = np.array(vecs).astype('float32')
        self.index.add(arr)
        if ids is None:
            start = len(self.meta['ids'])
            ids = list(range(start, start+len(docs)))
        self.meta['ids'].extend(ids)
        self.meta['docs'].extend(docs)
        self.save()

    def query(self, qvec, k=4):
        q = np.array([qvec]).astype('float32')
        D, I = self.index.search(q, k)
        results = []
        for idx in I[0]:
            if idx < len(self.meta['docs']):
                results.append(self.meta['docs'][idx])
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.meta, f)

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dim)
        self.meta = {"ids": [], "docs": []}
        self.save()

    def all_docs(self):
        return list(self.meta['docs'])

    def remove_doc(self, position):
        # 注意：這是簡單處理。完整做法需同步更新 index.
        # 這裡會重建 index 跟 meta
        if position < 0 or position >= len(self.meta['docs']):
            return False
        del self.meta['docs'][position]
        # 重新建立 index
        import numpy as np
        self.index = faiss.IndexFlatL2(self.dim)
        # 重新嵌入前端應該保存 embeddings, 但為簡單起見，這個範例會要求重新建立所有 embeddings
        self.save()
        return True
