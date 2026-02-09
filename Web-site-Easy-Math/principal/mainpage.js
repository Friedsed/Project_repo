



    const video1 = document.getElementById("myVideo1");

    function playPause() {
        if (video1.paused) {
            video1.play();
        } else {
            video1.pause();
        }
    }

    function skip(seconds) {
        video1.currentTime += seconds;
    }

    function setVolume(volume) {
        video1.volume = volume;
    }

    const video2 = document.getElementById("myVideo2");

    function playPause() {
        if (video2.paused) {
            video2.play();
        } else {
            video2.pause();
        }
    }

    function skip(seconds) {
        video2.currentTime += seconds;
    }

    function setVolume(volume) {
        video2.volume = volume;
    }

    const video3 = document.getElementById("myVideo3");

    function playPause() {
        if (video3.paused) {
            video3.play();
        } else {
            video3.pause();
        }
    }

    function skip(seconds) {
        video3.currentTime += seconds;
    }

    function setVolume(volume) {
        video3.volume = volume;
    }

    const video4 = document.getElementById("myVideo4");

    function playPause() {
        if (video4.paused) {
            video4.play();
        } else {
            video4.pause();
        }
    }

    function skip(seconds) {
        video4.currentTime += seconds;
    }

    function setVolume(volume) {
        video4.volume = volume;
    }

    const video5 = document.getElementById("myVideo5");

    function playPause() {
        if (video5.paused) {
            video5.play();
        } else {
            video5.pause();
        }
    }

    function skip(seconds) {
        video5.currentTime += seconds;
    }

    function setVolume(volume) {
        video5.volume = volume;
    }

    const video6 = document.getElementById("myVideo6");

    function playPause() {
        if (video6.paused) {
            video6.play();
        } else {
            video6.pause();
        }
    }

    function skip(seconds) {
        video6.currentTime += seconds;
    }

    function setVolume(volume) {
        video6.volume = volume;
    }


    document.getElementById("contactForm").addEventListener("submit", function(event) {
        event.preventDefault();
        
        const name = document.getElementById("name").value;
        const email = document.getElementById("email").value;
        const message = document.getElementById("message").value;
        
        // Envoyer l'e-mail (ici, vous devrez utiliser une API ou un service d'envoi d'e-mails)
                        // Exemple d'envoi d'e-mail en utilisant l'API SendGrid (côté serveur)
                const sgMail = require('friedlysedjro@gmail.com');
                sgMail.setApiKey('VOTRE_CLE_API_SENDGRID');

                const msg = {
                to: 'destinataire@example.com',
                from: 'expediteur@example.com',
                subject: 'Sujet de l\'e-mail',
                text: 'Contenu du message',
                html: '<p>Contenu du message en HTML</p>',
                };

                sgMail.send(msg)
                .then(() => {
                    console.log('E-mail envoyé avec succès');
                })
                .catch((error) => {
                    console.error('Erreur lors de l\'envoi de l\'e-mail', error);
                });



        // Réinitialiser le formulaire
        document.getElementById("contactForm").reset();
        alert("Votre message a été envoyé !");
      });
      
