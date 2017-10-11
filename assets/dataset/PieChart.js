import React from 'react';

import './PieChart.css';


class PieChart extends React.Component {

    drawD3(svgElement, data){
        var data = this.props.data,
            margins = {
                left: 10,
                bottom: 10,
                top: 10,
                right: 10,
            },
            h = $(svgElement).height() - margins.bottom - margins.top,
            w = $(svgElement).width() - margins.left - margins.right,
            radius = Math.min(h, w) / 2;

        var color = d3.scale.category10();

        var display_data = [];
        for (var key in data) {
            display_data.push({label: key, count: data[key]})
        }

        var g = d3.select(svgElement)
          .append('g')
          .attr('transform', 'translate(' + ((w + margins.left)/ 2) +
            ',' + ((h + margins.top)/ 2) + ')');

        var arc = d3.svg.arc()
          .innerRadius(0)
          .outerRadius(radius);

        var labelArc = d3.svg.arc()
            .outerRadius(radius - 40)
            .innerRadius(radius - 40);

        var pie = d3.layout.pie()
          .value(function(d) { return d.count; })
          .sort(null);

        g.selectAll('path')
          .data(pie(display_data))
          .enter()
          .append('path')
          .attr('d', arc)
          .style('stroke', 'black')
          .style('fill', function(d) {
              return color(d.data.label);
          });

    g.selectAll('text')
      .data(pie(display_data))
      .enter().append("text")
        .attr("transform", function(d) { return "translate(" + labelArc.centroid(d) + ")"; })
        .attr("dy", ".35em")
        .text(function(d) { return d.data.label + ": " + d.data.count; });
    }

    removeD3(svgElement){
        $(svgElement).empty();
    }

    drawPie(){
        var data = [{
            values: Object.values(this.props.data),
            labels: Object.keys(this.props.data),
            type: 'pie',
            hoverinfo: 'none',
        }];

        var layout = {
            height: $(this.refs.pie).width(),
            margin: {
                l: 10,
                r: 10,
                b: 10,
                t: 10,
                pad: 4,
            },
        };

        var options = {
            displaylogo: false,
            displayModeBar: false,
        };

        Plotly.newPlot(this.pieId(), data, layout, options);
    }

    clearPie(){
        $(this.refs.pie).empty();
    }

    componentDidMount(){
        this.drawPie();
    }

    componentWillUnmount(){
        this.clearPie();
    }

    pieId(){
        return '_pie_' + this.props.index;
    }

    render(){
        return <div ref='pie' id={this.pieId()}></div>;
    }
}

PieChart.defaultProps = {
    index: 0,
};

PieChart.propTypes = {
    data: React.PropTypes.object.isRequired,
    index: React.PropTypes.number.isRequired,
};

export default PieChart;
