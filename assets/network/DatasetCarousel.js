import React from 'react';

import './DatasetCarousel.css';

class DatasetCarousel extends React.Component {
  drawD3(carousel_inner, carousel_indicators, data_array){
    for (var i = 0; i < data_array.length; i++) {
      let data = data_array[i];
      let metaplot = data['metaplot'];
      let plot_name = data['assembly'] + ': ' + data['regions'];
      let margins = {
          left: 40,
          bottom: 60,
          top: 10,
          right: 10,
        },
        h = $(carousel_inner).height() - margins.bottom - margins.top,
        w = $(carousel_inner).width() - margins.left - margins.right;

      let $data_element = (i == 0) ?
        $('<div class="active item" style="text-align:center">' + plot_name + '</div>') :
        $('<div class="item" style="text-align:center">' + plot_name + '</div>');
      $(carousel_inner).append($data_element);

      let $indicator_element = (i == 0) ?
        $('<li data-target="#carouselExampleIndicators" data-slide-to=' + i + ' class="active"></li>') :
        $('<li data-target="#carouselExampleIndicators" data-slide-to=' + i + '></li>');
      $(carousel_indicators).append($indicator_element);

      let $svg_element = $('<svg style="height:100%; width:100%"></svg>');
      $data_element.append($svg_element);

      let window_values = [];
      for (var j in metaplot['bin_values']) {
          let val_1 = parseInt(metaplot['bin_values'][j][0]),
              val_2 = parseInt(metaplot['bin_values'][j][1]);
          window_values.push((val_1 + val_2)/2);
      }

      let scatter = [];
      for (let j = 0; j < window_values.length; j++) {
          scatter.push([window_values[j], metaplot['metaplot_values'][j]]);
      }

      let x = d3.scale.linear()
          .domain([metaplot['bin_values'][0][0],
            metaplot['bin_values'][metaplot['bin_values'].length-1][1]])
          .range([margins.left, w + margins.left]);

      let y = d3.scale.linear()
          .domain([0, d3.max(metaplot['metaplot_values'])])
          .range([h, margins.top])
          .nice()
          .clamp(true);

      let xAxis = d3.svg.axis()
          .scale(x)
          .ticks(3)
          .orient('bottom');

      let yAxis = d3.svg.axis()
          .scale(y)
          .ticks(4)
          .orient('left');

      d3.select($svg_element.get(0)).append('svg:g')
          .attr('class', 'x axis')
          .attr('transform', `translate(0,${$(carousel_inner).height() - margins.top - margins.bottom})`)
          .call(xAxis);

      d3.select($svg_element.get(0)).append('svg:g')
          .attr('class', 'y axis')
          .attr('transform', `translate(${margins.left},0)`)
          .call(yAxis);

      let line = d3.svg.line()
          .x(function(d) {return x(d[0])})
          .y(function(d) {return y(d[1])});

      d3.select($svg_element.get(0)).append('svg:path').attr('d', line(scatter));
    }
  }

  componentDidMount(){
      let carousel_inner = this.refs.carousel_inner;
      let carousel_indicators = this.refs.carousel_indicators;
      this.drawD3(carousel_inner, carousel_indicators, this.props.data);
  }

  render() {
    let carousel_id = 'metaplot_carousel_' + this.props.id;
    return <div id={carousel_id} className="carousel slide" data-ride="carousel" data-interval="false" style={{height: '100%', width: '100%'}}>
      <ol ref='carousel_indicators' className="carousel-indicators"></ol>
      <div ref='carousel_inner' className="carousel-inner" role="listbox" style={{height: '100%', width: '80%', left: '10%'}}>
      </div>
      <a className="left carousel-control" href={'#' + carousel_id} role="button" data-slide="prev">
        <span className="glyphicon glyphicon-chevron-left" aria-hidden="true"></span>
        <span className="sr-only">Previous</span>
      </a>
      <a className="right carousel-control" href={'#' + carousel_id} role="button" data-slide="next">
        <span className="glyphicon glyphicon-chevron-right" aria-hidden="true"></span>
        <span className="sr-only">Next</span>
      </a>
    </div>;
  }
};

DatasetCarousel.propTypes = {
  data: React.PropTypes.array.isRequired,
  id: React.PropTypes.number,
};

DatasetCarousel.defaultProps = {
  id: 0,
};

export default DatasetCarousel;
