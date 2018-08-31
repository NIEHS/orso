import React from 'react';

import './MetaPlot.css';


class MetaPlot extends React.Component {

    drawD3(svgElement, data){
        var margins = {
                left: 40,
                bottom: 60,
                top: 10,
                right: 10,
            },
            h = $(svgElement).height() - margins.bottom - margins.top,
            w = $(svgElement).width() - margins.left - margins.right;

        var window_values = [];
        for (var i in data['bin_values']) {
            var val_1 = parseInt(data['bin_values'][i][0]),
                val_2 = parseInt(data['bin_values'][i][1]);
            window_values.push((val_1 + val_2)/2);
        }

        var scatter = [];
        for (var i = 0; i < window_values.length; i++) {
            scatter.push([window_values[i], data['metaplot_values'][i]]);
        }

        var x = d3.scale.linear()
            .domain([data['bin_values'][0][0],data['bin_values'][data['bin_values'].length-1][1]])
            .range([margins.left, w + margins.left]);

        var y = d3.scale.linear()
            .domain([0, d3.max(data['metaplot_values'])])
            .range([h, margins.top])
            .nice()
            .clamp(true);

        var xAxis = d3.svg.axis()
            .scale(x)
            .ticks(3)
            .orient('bottom');

        var yAxis = d3.svg.axis()
            .scale(y)
            .ticks(4)
            .orient('left');

        d3.select(svgElement).append('svg:g')
            .attr('class', 'x axis')
            .attr('transform', `translate(0,${($(svgElement).height() - margins.top - margins.bottom)})`)
            .call(xAxis);

        d3.select(svgElement).append('svg:g')
            .attr('class', 'y axis')
            .attr('transform', `translate(${margins.left},0)`)
            .call(yAxis);

        var line = d3.svg.line()
            .x(function(d) {return x(d[0])})
            .y(function(d) {return y(d[1])});

        d3.select(svgElement).append('svg:path').attr('d', line(scatter));
    }

    removeD3(svgElement){
        $(svgElement).empty();
    }

    componentDidMount(){
        let svg = this.refs.svg;
        this.drawD3(svg, this.props.data);
    }

    componentWillUnmount(){
        this.removeD3(this.refs.svg);
    }

    render(){
        return <div className='metaplot'>
            <svg style={{height:"100%", width:"100%"}} ref='svg'></svg>
        </div>;
    }
}

MetaPlot.propTypes = {
    data: React.PropTypes.object.isRequired,
};

export default MetaPlot;
