
$(document).ready(function(){
	console.log('Document is ready')

	// Button config for Linear Regression Prediction
	$("#refresh").click(async function(){
		console.log('button was clicked')


		const response = await $.ajax('/prediction', {
			method: "post",
			contentType:'application/json'
		})

		console.log(response)
		$('#prediction').val(response.prediction)

	})

})








