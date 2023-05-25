def big_chart(chart, fontsize = 20): 
    return chart.configure_axis(
            grid = False, 
        labelFontSize = fontsize,
        titleFontSize = fontsize, 
            offset = 5, 
    ).configure_title(
        fontSize = fontsize
        ).configure_legend(
    titleFontSize=fontsize,
    labelFontSize=fontsize
    ).configure_view(
        strokeWidth=0
)