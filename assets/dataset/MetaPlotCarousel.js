import React from 'react';


class MetaPlotCarousel extends React.Component {

    drawMetaplot(plot_data){
        var div_number = $(this.refs.carousel_inner).children('div').length,
            plot_name = plot_data['assembly'] + ': ' + plot_data['regions'];

        let $data_element = (div_number == 0) ?
            $('<div id="' + this.props.exp_id + '.' + div_number + '" class="active item" style="text-align:center">' + plot_name + '</div>') :
            $('<div id="' + this.props.exp_id + '.' + div_number + '" class="item" style="text-align:center">' + plot_name + '</div>');
        $(this.refs.carousel_inner).append($data_element);

        let $indicator_element = (div_number == 0) ?
            $('<li data-target="#carouselExampleIndicators" data-slide-to=' + div_number + ' class="active"></li>') :
            $('<li data-target="#carouselExampleIndicators" data-slide-to=' + div_number + '></li>');
        $(this.refs.carousel_indicators).append($indicator_element);

        var x = [], y = [];

        for (var i = 0; i < plot_data['metaplot']['bin_values'].length; i++) {
            x.push(plot_data['metaplot']['bin_values'][i]);
            y.push(plot_data['metaplot']['metaplot_values'][i]);
        }

        var data = [{
            x: x,
            y: y,
            type: 'scatter',
        }];

        var layout = {
            autosize: false,
            height: $(this.refs.carousel_inner).height(),
            width: $(this.refs.carousel_inner).width(),
            xaxis: {
                tickvals: plot_data['metaplot']['ticks']['tickvals'],
                ticktext: plot_data['metaplot']['ticks']['ticktext'],
            },
        };

        Plotly.newPlot('' + this.props.exp_id + '.' + div_number, data, layout);
    }

    componentDidMount(){
        for (var i = 0; i < this.props.data.length; i++) {
            this.drawMetaplot(this.props.data[i]);
        }
    }

    componentWillUnmount(){
        $(this.refs.carousel).clear();
    }

    render(){
        return <div ref='carousel' id='_carousel' className='carousel slide' data-ride='carousel' data-interval='false' style={{height: '100%', width: '100%'}}>
            <ol ref='carousel_indicators' className='carousel-indicators'></ol>
            <div ref='carousel_inner' className='carousel-inner' role='listbox' style={{height: '100%', width: '80%', left: '10%'}}></div>
            <a className='left carousel-control' href='#_carousel' role='button' data-slide='prev'>
                <span className='glyphicon glyphicon-chevron-left' aria-hidden='true'></span>
                <span className='sr-only'>Previous</span>
            </a>
            <a className='right carousel-control' href='#_carousel' role='button' data-slide='next'>
                <span className='glyphicon glyphicon-chevron-right' aria-hidden='true'></span>
                <span className='sr-only'>Next</span>
            </a>
        </div>;
    }
}

MetaPlotCarousel.propTypes = {
    data: React.PropTypes.array.isRequired,
};

export default MetaPlotCarousel;
