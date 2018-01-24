import React from 'react';


class RecommendationScatter extends React.Component {

    constructor(props) {
        super(props);

        var exp_type_choices = (['--']).concat(Object.keys(this.props.data['paired_data']));

        this.state = {
            exp_type: '--',
            exp_type_choices: exp_type_choices,
        };
    }

    datasetUrl(pk_1, pk_2) {
        return `/network/dataset-comparison/${pk_1}-${pk_2}/`;
    }

    drawPlotlyScatter(){
        var plot = document.getElementById('rec_scatter_plot');
        var data = this.props.data['paired_data'][this.state.exp_type];

        var x = [], y = [], names = [];

        if (this.state.exp_type != '--') {
            for(var i = 0; i < data.length; i++){
                names.push(data[i][1] + ':' + data[i][3])
                x.push(data[i][4]);
                y.push(data[i][5]);
            }
        }

        var trace_1 = {
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            text: names,
            marker: {
                opacity: 0.4
            },
            point_data: data,
        }

        var plot_data = [trace_1];

        var layout = {
            xaxis: {
                autorange: true,
            },
            yaxis: {
                autorange: true,
            },
        };

        Plotly.newPlot('rec_scatter_plot', plot_data, layout);

        plot.on('plotly_click', function(data){
            for(var i = 0; i < data.points.length; i++){
                var index = data.points[i].pointNumber;
                var pk_1 = data.points[i].data.point_data[index][0],
                    pk_2 = data.points[i].data.point_data[index][2];
            }
            window.open(this.datasetUrl(pk_1, pk_2));
        }.bind(this));
    }

    drawPlotlyBoxPlot(){
        var plot_data = [];

        if (this.state.exp_type != '--') {
            var _data = this.props.data['quartiled_data'][this.state.exp_type];
            for(var i = 0; i < _data.length; i++){
                plot_data.push({
                    y: _data[i],
                    type: 'box',
                    boxpoints: 'all',
                    jitter: 0.3,
                    pointpos: -1.8,
                });
            }
        }

        Plotly.newPlot('rec_box_plot', plot_data);
    }

    drawPlotly(){
        this.drawPlotlyScatter();
        this.drawPlotlyBoxPlot();
    }

    componentDidMount(){
        this.drawPlotly();

        for (let i in this.state.exp_type_choices) {
            $(this.refs.exp_type_select).append(
                '<option val="' + i + '">' + this.state.exp_type_choices[i] + '</option>');
        }
    }

    componentWillUnmount(){
        $(this.refs.gene_scatter_plot).clear();
    }

    update_exp_type(event){
        this.setState({
            exp_type: event.target.value,
        }, this.drawPlotly);
    }

    render(){
        return <div>
            <select ref='exp_type_select'
                onChange={this.update_exp_type.bind(this)}
                value={this.state.exp_type}>
            </select>
            <div ref='rec_scatter_plot' id='rec_scatter_plot'></div>
            <div ref='rec_box_plot' id='rec_box_plot'></div>
        </div>;
    }
}

RecommendationScatter.propTypes = {
    data: React.PropTypes.object.isRequired,
};

export default RecommendationScatter;
