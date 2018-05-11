import React from 'react';


class Dendrogram extends React.Component {

    drawDendrogram(){
        var data = this.props.dendrogram['data'];
        var layout = this.props.dendrogram['layout'];

        var lines = layout['shapes'];
        var tickvals = [];
        var y_max = 0;

        for (var i = 0; i < lines.length; i++) {
            if (lines[i]['y0'] > y_max) {
                y_max = lines[i]['y0'];
            }
        }

        var step = Math.ceil(y_max / 10);
        for (var i = 0; i < y_max; i = i + step) {
            tickvals.push(i);
        }

        layout['xaxis'] = {
            showgrid: false,
            zeroline: false,
            showline: false,
            showticklabels: false,
        };

        layout['yaxis'] = {
            zeroline: false,
            tickvals: tickvals,
        };

        Plotly.newPlot('dendrogram', data, layout);
    }

    clearDendrogram(){
        $(this.refs.dendrogram).empty();
    }

    componentDidMount(){
        this.drawDendrogram();
    }

    componentWillUnmount(){
        this.clearDendrogram();
    }

    render(){
        return <div style={{border: '1px solid black'}}>
            <div
                style={{height:'100%', width:"100%"}}
                ref='dendrogram' id='dendrogram'>
            </div>
        </div>;
    }
}

Dendrogram.propTypes = {
    dendrogram: React.PropTypes.object.isRequired,
};

export default Dendrogram;
