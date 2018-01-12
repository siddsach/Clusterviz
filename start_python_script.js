
var fs = require('fs');

module.exports = function(votes, done) {
  //console.log('\n\n\n\n')
  //console.log("votes: " + JSON.stringify(votes));
  //console.log('\n\n\n\n\n\n')
  var spawn = require('child_process').spawn
  var py = spawn('python3', ['assets/python-scripts/ml_script.py'])

  var dataString = '';
  var vis_data

  py.stdout.on('data', function(data){
    dataString += data.toString();
  });

  py.stdout.on('end', function(){
    if (dataString == '') {
      dataString = '{}'
    }
    //console.log("dataString: " + dataString);
    var vis_data = '{}';
    try {
       vis_data = JSON.parse(dataString);
    } catch (err) {
      console.log(err);
    }
    save(JSON.stringify(vis_data), done);
  });

  //Pipes Python's stderr to Node stdout for debugging
  py.stderr.pipe(process.stdout);

  //Input JSON goes here with following format
  var data = votes;
  py.stdin.write(JSON.stringify(data));
  py.stdin.end();
}

function save(data, cb) {
  cb(data);
}
