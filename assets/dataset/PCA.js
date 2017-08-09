import React from 'react';


class PCA extends React.Component {

    constructor(props) {
        super(props);

        let color_by_choices = ['None', 'Cell type', 'Target'];

        this.state = {
            color_by: 'None',
            color_by_choices: color_by_choices,
        };
    }

    drawPlotly(divElement, data, experiment_urls) {
        var pca = [], cell_types = [], targets = [], names = [], urls = [];
        for (var i = 0; i < data.length; i++) {
            pca[i] = {
                x: data[i]['transformed_values'][0],
                y: data[i]['transformed_values'][1],
                z: data[i]['transformed_values'][2],
            };
            cell_types[i] = data[i]['experiment_cell_type'];
            targets[i] = data[i]['experiment_target'];
            urls[i] = experiment_urls[data[i]['experiment_pk']];
            if (data[i]['experiment_target'] == '') {
                names[i] = `Cell type: ${data[i]['experiment_cell_type']}`;
            } else {
                names[i] = `Cell type: ${data[i]['experiment_cell_type']}
                            <br>Target: ${data[i]['experiment_target']}`;
            }
        }

        var color_scale = d3.scale.category20c();
        var colors = [];
        if (this.state.color_by == 'None') {
            for (var i = 0; i < pca.length; i++) {
                colors.push(color_scale(0));
            }
        } else {
            if (this.state.color_by == 'Cell type') {
                var list = cell_types;
            } else if (this.state.color_by == 'Target') {
                var list = targets;
            }
            let used = [];
            for (var i = 0; i < list.length; i++) {
                if ( $.inArray(list[i], used) == -1) {
                    used.push(list[i]);
                }
                colors.push(color_scale($.inArray(list[i], used) % 20));
            }
        }

        function unpack(rows, key) {
            return rows.map(function(row)
                { return row[key]; });
        }
        var trace1 = {
            x: unpack(pca, 'x'),
            y: unpack(pca, 'y'),
            z: unpack(pca, 'z'),
            mode: 'markers',
            text: names,
            url: urls,
            hoverinfo: 'text',
            marker: {
                size: 12,
                color: colors,
                opacity: 0.8,
            },
            type: 'scatter3d',
        };

        var data = [trace1];
        var layout = {
            margin: {
                l: 0,
                r: 0,
                b: 0,
                t: 0,
            },
            scene: {
        		xaxis:{
                    title: '',
                    showticklabels: false,
                },
        		yaxis:{
                    title: '',
                    showticklabels: false,
                },
        		zaxis:{
                    title: '',
                    showticklabels: false,
                },
    		},
        };
        var options = {
            displaylogo: false,
            modeBarButtonsToRemove: [
                'sendDataToCloud',
                'resetCameraLastSave3d',
                'hoverClosest3d',
            ],
        };
        Plotly.newPlot('plot', data, layout, options);

        var plot = document.getElementById('plot');
        plot.on('plotly_click', function(data){
            var url = '',
                name = '';
            for(var i = 0; i < data.points.length; i++){
                var index = data.points[i].pointNumber;
                url = data.points[i].data.url[index];
                name = data.points[i].text;
            }
            window.open(window.location.href.replace(window.location.pathname, url), name)
        });
    }

    removePlotly(divElement){
        $(divElement).empty();
    }

    componentDidMount(){
        var $experiment_select = $(this.refs.select_experiment_type);
        for (let i in this.state.experiment_type_choices) {
            $experiment_select.append(
                '<option val="' + i + '">' + this.state.experiment_type_choices[i] + '</option>');
        }

        var $color_by_select = $(this.refs.color_by_select);
        for (let i in this.state.color_by_choices) {
            $color_by_select.append(
                '<option val="' + i + '">' + this.state.color_by_choices[i] + '</option>');
        }
        console.log(this.props.data);
        this.drawPlotly(this.refs.plot, this.props.data, this.props.exp_urls);
    }

    componentDidUpdate(){
        this.removePlotly(this.refs.plot);
        this.drawPlotly(this.refs.plot, this.props.data, this.props.exp_urls);
    }

    componentWillUnmount(){
        this.removeD3(this.refs.svg);
    }

    change_color(event){
         this.setState({color_by: event.target.value});
    }

    render(){
        return <div className='pca'>
            <div className='row'>
                <div className='col-sm-8'>
                    <div ref='plot' id='plot'></div>
                </div>
                <div className='col-sm-4'>
                    <div>Color by</div>
                    <select ref='color_by_select'
                        onChange={this.change_color.bind(this)}
                        value={this.state.color_by}>
                    </select>
                </div>
            </div>
        </div>;
    }
}

PCA.propTypes = {
    data: React.PropTypes.array.isRequired,
    exp_urls: React.PropTypes.object.isRequired,
};

export default PCA;
