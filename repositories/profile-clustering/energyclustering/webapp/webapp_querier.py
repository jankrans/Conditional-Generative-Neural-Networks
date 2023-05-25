

class DummyQuerier():
    def __init__(self, log_path=None):
        self.log_path = log_path
    def update_clustering(self, x):
        pass

    def continue_cluster_process(self):
        return True

    def query_points(self, i1, i2):
        result = yield (i1,i2)
        if self.log_path is not None:
            with self.log_path.open(mode = 'a+') as file:
                file.write(f"{i1}, {i2}, {result}\n")
        return result

#
# class WebappQuerier():
#     def __init__(self):
#         super().__init__()
#         self.webapp = None
#
#     def query_points(self, idx1, idx2):
#         if self.webapp is None:
#             # init webapp
#             self.webapp = WebApp(ClusteringProfileVisualizer, idx1, idx2)
#             self.webapp.run()
#         else:
#             self.webapp.new_query(idx1,idx2)
#         query_result = yield
#
#         pass
#
# if __name__ == '__main__':
#     cobras = DummyCOBRAS()
#     cobras.cluster()
