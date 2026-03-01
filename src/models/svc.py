from sklearn.svm import SVC

def svc(C=1.0, kernel="linear", degree=3, **kwargs):
    # Only relevant for kernel="poly"
    if kernel == "poly":
        return SVC(C=C, kernel=kernel, degree=degree, **kwargs)
    return SVC(C=C, kernel=kernel, **kwargs)