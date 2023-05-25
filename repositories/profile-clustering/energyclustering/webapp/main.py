from energyclustering.webapp.profilevis import HeatmapProfileVisualizer, SimpleProfileVisualiser, DayProfileVisualizer
from energyclustering.webapp.profileviscomposite import ProfileVisualizerDropbox
from energyclustering.webapp.webapp import WebApp

if __name__ == '__main__':
    # WebApp(ClusteringProfileVisualizer).run()
    # WebApp(HeatmapProfileVisualizer).run()
    WebApp(lambda x, data: ProfileVisualizerDropbox(x, ['heatmap',  'daily_line', 'simple'], [ HeatmapProfileVisualizer, DayProfileVisualizer, SimpleProfileVisualiser], data)).run()